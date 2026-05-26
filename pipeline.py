"""
Pipeline Parallelism for the Encoder-Decoder Transformer.

This module implements manual pipeline parallelism by splitting the Transformer
into two stages:

  Stage 0 (Encoder Stage):
      src_embed -> src_pos -> Encoder
      Produces: encoder_output (batch, seq_len, d_model)

  Stage 1 (Decoder Stage):
      tgt_embed -> tgt_pos -> Decoder -> ProjectionLayer
      Consumes: encoder_output from Stage 0
      Produces: logits (batch, seq_len, vocab_size)

The pipeline schedule splits each batch into micro-batches and streams them
through the stages.  A simple "all-forward-then-all-backward" (AFAB) schedule
is used for clarity and debuggability.

Communication between stages uses point-to-point send/recv on the PP process
group extracted from the 3D DeviceMesh.

Architecture reminder (user's target topology):
    DP=2 clusters  x  PP=2 nodes/cluster  x  TP=2 GPUs/node  =  8 GPUs total
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------

class EncoderStage(nn.Module):
    """
    Pipeline Stage 0: embedding + positional encoding + encoder stack.

    forward(src, src_mask) -> encoder_output
    """

    def __init__(self, src_embed, src_pos, encoder):
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.encoder = encoder

    def forward(self, src, src_mask):
        x = self.src_embed(src)
        x = self.src_pos(x)
        return self.encoder(x, src_mask)


class DecoderStage(nn.Module):
    """
    Pipeline Stage 1: embedding + positional encoding + decoder stack + projection.

    forward(encoder_output, src_mask, tgt, tgt_mask) -> logits
    """

    def __init__(self, tgt_embed, tgt_pos, decoder, projection_layer):
        super().__init__()
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.decoder = decoder
        self.projection_layer = projection_layer

    def forward(self, encoder_output, src_mask, tgt, tgt_mask):
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return self.projection_layer(x)


# ---------------------------------------------------------------------------
# Model splitting
# ---------------------------------------------------------------------------

def split_model_into_stages(transformer, pp_rank, device):
    """
    Split a Transformer into its pipeline stage for the given *pp_rank*.

    Args:
        transformer: a fully-constructed ``Transformer`` instance (from model.py).
        pp_rank:     0 or 1.
        device:      torch device to place the stage on.

    Returns:
        nn.Module — the stage submodule, already on *device*.
    """
    if pp_rank == 0:
        stage = EncoderStage(
            transformer.src_embed,
            transformer.src_pos,
            transformer.encoder,
        )
    elif pp_rank == 1:
        stage = DecoderStage(
            transformer.tgt_embed,
            transformer.tgt_pos,
            transformer.decoder,
            transformer.projection_layer,
        )
    else:
        raise ValueError(f"pp_rank must be 0 or 1, got {pp_rank}")

    return stage.to(device)


# ---------------------------------------------------------------------------
# Micro-batch helpers
# ---------------------------------------------------------------------------

def _chunk_batch(batch_dict, num_microbatches):
    """
    Split every tensor in *batch_dict* along the batch dimension (dim 0)
    into *num_microbatches* chunks.  Non-tensor values (like src_text) are
    dropped since they are only needed for validation, not training.

    Returns a list of dicts, one per micro-batch.
    """
    keys_to_chunk = [
        "encoder_input", "decoder_input",
        "encoder_mask", "decoder_mask",
        "label",
    ]
    chunks = [{} for _ in range(num_microbatches)]
    for key in keys_to_chunk:
        if key in batch_dict:
            tensors = batch_dict[key].chunk(num_microbatches, dim=0)
            for i, t in enumerate(tensors):
                chunks[i][key] = t
    return chunks


# ---------------------------------------------------------------------------
# Pipeline schedule  (All-Forward-All-Backward)
# ---------------------------------------------------------------------------

def pipeline_forward_backward(
    stage,
    pp_rank,
    pp_group,
    pp_world_size,
    micro_batches,
    loss_fn,
    device,
    d_model,
    seq_len,
):
    """
    Run the full-batch pipeline schedule for one training step.

    This implements a simple All-Forward-All-Backward (AFAB) schedule:
      1.  All micro-batches run forward through stage 0, then stage 1.
      2.  All micro-batches run backward through stage 1, then stage 0.

    Args:
        stage:          The local nn.Module for this PP rank.
        pp_rank:        0 (encoder) or 1 (decoder).
        pp_group:       The torch ProcessGroup for the PP dimension.
        pp_world_size:  Number of PP stages (always 2 for now).
        micro_batches:  List of dicts, one per micro-batch.
        loss_fn:        Loss function (only used on the last stage).
        device:         torch device.
        d_model:        Model hidden dimension.
        seq_len:        Sequence length.

    Returns:
        total_loss (float) — accumulated loss over micro-batches
                             (meaningful only on the last PP stage).
    """
    num_mb = len(micro_batches)
    total_loss = 0.0

    assert pp_world_size <= 2, "Pipeline schedule currently strictly supports exactly 2 stages (Encoder and Decoder)."
    # Determine peer rank within the PP group
    # Stage 0 sends to stage 1, stage 1 receives from stage 0
    peer_rank = 1 if pp_rank == 0 else 0

    # Storage for activations needed for backward
    saved_inputs = []       # encoder_output tensors (need grad)
    saved_outputs = []      # stage outputs (for backward)

    # -----------------------------------------------------------------------
    # FORWARD PASS — all micro-batches
    # -----------------------------------------------------------------------
    for mb_idx in range(num_mb):
        mb = micro_batches[mb_idx]
        mb_batch_size = mb["encoder_input"].shape[0]

        if pp_rank == 0:
            # --- Stage 0: Encoder ---
            src = mb["encoder_input"].to(device)
            src_mask = mb["encoder_mask"].to(device)

            encoder_output = stage(src, src_mask)  # (mb_batch, seq_len, d_model)

            # Send encoder_output to Stage 1
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Sending encoder_output to peer {peer_rank}...")
            dist.send(encoder_output.contiguous(), group=pp_group, group_dst=peer_rank)
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Successfully sent encoder_output.")

            # Save for backward (we need to backprop through encoder_output)
            saved_outputs.append(encoder_output)

        else:
            # --- Stage 1: Decoder ---
            tgt = mb["decoder_input"].to(device)
            tgt_mask = mb["decoder_mask"].to(device)
            src_mask = mb["encoder_mask"].to(device)
            label = mb["label"].to(device)

            # Receive encoder_output from Stage 0
            encoder_output = torch.zeros(
                mb_batch_size, seq_len, d_model, device=device
            )
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Waiting to receive encoder_output from peer {peer_rank}...")
            dist.recv(encoder_output, group=pp_group, group_src=peer_rank)
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Successfully received encoder_output.")
            encoder_output.requires_grad_(True)  # need grad to send back to stage 0

            logits = stage(encoder_output, src_mask, tgt, tgt_mask)

            # Compute loss on last stage
            vocab_size = logits.shape[-1]
            loss = loss_fn(logits.view(-1, vocab_size), label.view(-1))
            total_loss += loss.item()

            # Save for backward
            saved_inputs.append(encoder_output)
            saved_outputs.append(loss)

    # -----------------------------------------------------------------------
    # BACKWARD PASS — all micro-batches (reverse order)
    # -----------------------------------------------------------------------
    for mb_idx in reversed(range(num_mb)):
        if pp_rank == 1:
            # --- Stage 1 backward ---
            loss = saved_outputs[mb_idx]
            encoder_output = saved_inputs[mb_idx]

            loss.backward()

            # Guard: if grad is None the decoder's cross-attention did not
            # flow gradients back through encoder_output (broken compute graph).
            if encoder_output.grad is None:
                raise RuntimeError(
                    f"[PP Rank 1 | MB {mb_idx}] encoder_output.grad is None after "
                    "backward. Cross-attention must use encoder_output in the "
                    "forward pass and requires_grad_(True) must be set before it."
                )
            # Send encoder_output.grad back to Stage 0
            grad = encoder_output.grad.contiguous()
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Sending grad to peer {peer_rank}...")
            dist.send(grad, group=pp_group, group_dst=peer_rank)
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Successfully sent grad.")

        else:
            # --- Stage 0 backward ---
            encoder_output = saved_outputs[mb_idx]
            mb = micro_batches[mb_idx]
            mb_batch_size = mb["encoder_input"].shape[0]

            # Receive gradient from Stage 1
            grad = torch.zeros(
                mb_batch_size, seq_len, d_model, device=device
            )
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Waiting to receive grad from peer {peer_rank}...")
            dist.recv(grad, group=pp_group, group_src=peer_rank)
            if os.environ.get("DEBUG_PP") == "1":
                print(f"[PP Rank {pp_rank} | MB {mb_idx}] Successfully received grad.")

            # Backward through encoder
            encoder_output.backward(grad)

    return total_loss
