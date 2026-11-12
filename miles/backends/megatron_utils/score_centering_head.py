"""Use Megatron's output processor to bypass full-sequence vocabulary logits."""

from miles.backends.training_utils.loss.hub.logit_processors import _iter_response_chunks
from miles.backends.training_utils.loss.hub.score_centering.chunked_head import chunked_head_log_probs
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_from_hidden


class ChunkedScoreHead:
    """Own the output-head parameters for one microbatch's forward/loss callback."""

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self.weight = None
        self.bias = None

    def capture(self, *, hidden_states, output_layer, output_weight, config, **kwargs):
        if getattr(config, "use_mup", False):
            raise ValueError("Chunked score head requires an unscaled affine output projection")
        self.weight = output_weight if output_weight is not None else output_layer.weight
        self.bias = output_layer.bias
        # Match GPTModel's normal output layout [sequence, batch, hidden] -> [batch, sequence, hidden].
        return hidden_states.transpose(0, 1).contiguous()

    def loss(self, args, batch, hidden, reduce):
        return score_centering_loss_from_hidden(
            args, batch, hidden, reduce, weight=self.weight, bias=self.bias, head_chunk_size=self.chunk_size
        )

    def log_probs(
        self, hidden, *, args, unconcat_tokens, total_lengths, response_lengths, max_seq_lens=None, with_entropy=False
    ):
        """Reference/actor forward-only scoring uses the same bounded projection."""
        result = {"log_probs": []}
        if with_entropy:
            result["entropy"] = []
        for rows, tokens, _ in _iter_response_chunks(
            hidden,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            include_response_indices=False,
        ):
            selected, entropy = chunked_head_log_probs(
                rows,
                tokens[:, None],
                weight=self.weight,
                bias=self.bias,
                head_chunk_size=self.chunk_size,
                vocab_size=args.vocab_size,
                temperature=args.rollout_temperature,
                chunk_size=args.log_probs_chunk_size,
                with_entropy=with_entropy,
            )
            result["log_probs"].append(selected[:, 0].detach())
            if with_entropy:
                result["entropy"].append(entropy.detach())
        return result
