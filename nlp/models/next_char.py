from typing import Dict, Tuple

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import SoftmaxLoss
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import Perplexity

@Model.register('namegen')
class NameGen(Model):
    """ The `NextTokenLM` embeds some input tokens, contextualizes them, ,then predicts the next work
        If `BeamSearch` is given, this model will predict a sequence of tokens
    """
    def __init__(self, 
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            contextualizer: Seq2SeqEncoder = None,
            dropout:float = 0.0,
            num_samples:int = None,
            sparse_embeddings:bool=False,
            bidirectional:bool=False,
            initializer=InitializerApplicator(),
            **kwargs
            )->None:
        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._contextualizer = contextualizer
        self._bidirectional = bidirectional

        if self._bidirectional:
            self._forward_dim = contextualizer.get_output_dim() // 2
        else:
            self._forward_dim = contextualizer.get_output_dim()

        self._softmax_loss = SoftmaxLoss(
            num_words = vocab.get_vocab_size(), embedding_dim = self._forward_dim
        )


        self._perplexity = Perplexity()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = lambda x: x

        
        if initializer is not None:
            initializer(self)


    def _get_target_token_embeddings(
        self, token_embeddings: torch.Tensor, mask: torch.BoolTensor, direction: int
    ) -> torch.Tensor:
        # Need to shift the mask in the correct direction
        zero_col = token_embeddings.new_zeros(mask.size(0), 1).to(dtype=torch.bool)
        if direction == 0:
            # forward direction, get token to right
            shifted_mask = torch.cat([zero_col, mask[:, 0:-1]], dim=1)
        else:
            shifted_mask = torch.cat([mask[:, 1:], zero_col], dim=1)
        return token_embeddings.masked_select(shifted_mask.unsqueeze(-1)).view(
            -1, self._forward_dim
        )

    def _compute_loss(
        self,
        lm_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        forward_targets: torch.Tensor,
        backward_targets: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If bidirectional, lm_embeddings is shape (batch_size, timesteps, dim * 2)
        # If unidirectional, lm_embeddings is shape (batch_size, timesteps, dim)
        # forward_targets, backward_targets (None in the unidirectional case) are
        # shape (batch_size, timesteps) masked with 0
        if self._bidirectional:
            forward_embeddings, backward_embeddings = lm_embeddings.chunk(2, -1)
            backward_loss = self._loss_helper(
                1, backward_embeddings, backward_targets, token_embeddings
            )
        else:
            forward_embeddings = lm_embeddings
            backward_loss = None

        forward_loss = self._loss_helper(0, forward_embeddings, forward_targets, token_embeddings)
        return forward_loss, backward_loss

    def _loss_helper(
        self,
        direction: int,
        direction_embeddings: torch.Tensor,
        direction_targets: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> Tuple[int, int]:
        mask = direction_targets > 0
        # we need to subtract 1 to undo the padding id since the softmax
        # does not include a padding dimension

        # shape (batch_size * timesteps, )
        non_masked_targets = direction_targets.masked_select(mask) - 1

        # shape (batch_size * timesteps, embedding_dim)
        non_masked_embeddings = direction_embeddings.masked_select(mask.unsqueeze(-1)).view(
            -1, self._forward_dim
        )
        # note: need to return average loss across forward and backward
        # directions, but total sum loss across all batches.
        # Assuming batches include full sentences, forward and backward
        # directions have the same number of samples, so sum up loss
        # here then divide by 2 just below
        if not self._softmax_loss.tie_embeddings or not self._use_character_inputs:
            return self._softmax_loss(non_masked_embeddings, non_masked_targets)
        else:
            # we also need the token embeddings corresponding to the
            # the targets
            raise NotImplementedError(
                "This requires SampledSoftmaxLoss, which isn't implemented yet."
            )

            non_masked_token_embeddings = self._get_target_token_embeddings(
                token_embeddings, mask, direction
            )
            return self._softmax(
                non_masked_embeddings, non_masked_targets, non_masked_token_embeddings
            )

    def delete_softmax(self) -> None:
        """
        Remove the softmax weights. Useful for saving memory when calculating the loss
        is not necessary, e.g. in an embedder.
        """
        self._softmax_loss = None

    def num_layers(self) -> int:
        """
        Returns the depth of this LM. That is, how many layers the contextualizer has plus one for
        the non-contextual layer.
        """
        if hasattr(self._contextualizer, "num_layers"):
            return self._contextualizer.num_layers + 1
        else:
            raise NotImplementedError(
                f"Contextualizer of type {type(self._contextualizer)} "
                + "does not report how many layers it has."
            )

    def forward(self, tokens: TextFieldTensors) -> Dict[str, torch.Tensor]:  # type: ignore
        source = tokens
        """
        Computes the averaged forward (and backward, if language model is bidirectional)
        LM loss from the batch.
        # Parameters
        source : `TextFieldTensors`, required.
            The output of `Batch.as_tensor_dict()` for a batch of sentences. By convention,
            it's required to have at least a `"tokens"` entry that's the output of a
            `SingleIdTokenIndexer`, which is used to compute the language model targets.
        # Returns
        Dict with keys:
        `'loss'` : `torch.Tensor`
            forward negative log likelihood, or the average of forward/backward
            if language model is bidirectional
        `'forward_loss'` : `torch.Tensor`
            forward direction negative log likelihood
        `'backward_loss'` : `torch.Tensor` or `None`
            backward direction negative log likelihood. If language model is not
            bidirectional, this is `None`.
        `'lm_embeddings'` : `Union[torch.Tensor, List[torch.Tensor]]`
            (batch_size, timesteps, embed_dim) tensor of top layer contextual representations or
            list of all layers. No dropout applied.
        `'noncontextual_token_embeddings'` : `torch.Tensor`
            (batch_size, timesteps, token_embed_dim) tensor of bottom layer noncontextual
            representations
        `'mask'` : `torch.BoolTensor`
            (batch_size, timesteps) mask for the embeddings
        """

        mask = get_text_field_mask(source)

        # shape (batch_size, timesteps, embedding_size)
        embeddings = self._text_field_embedder(source)

        # Either the top layer or all layers.
        contextual_embeddings: Union[torch.Tensor, List[torch.Tensor]] = self._contextualizer(
            embeddings, mask
        )

        return_dict = {}

        # If we have target tokens, calculate the loss.
        token_id_dict = source.get("tokens")
        if token_id_dict is not None:
            token_ids = token_id_dict["tokens"]
            assert isinstance(contextual_embeddings, torch.Tensor)

            # Use token_ids to compute targets
            forward_targets = torch.zeros_like(token_ids)
            forward_targets[:, 0:-1] = token_ids[:, 1:]

            if self._bidirectional:
                backward_targets = torch.zeros_like(token_ids)
                backward_targets[:, 1:] = token_ids[:, 0:-1]
            else:
                backward_targets = None

            # add dropout
            contextual_embeddings_with_dropout = self._dropout(contextual_embeddings)

            # compute softmax loss
            forward_loss, backward_loss = self._compute_loss(
                contextual_embeddings_with_dropout, embeddings, forward_targets, backward_targets
            )

            num_targets = torch.sum((forward_targets > 0).long())
            if num_targets > 0:
                if self._bidirectional:
                    average_loss = 0.5 * (forward_loss + backward_loss) / num_targets.float()
                else:
                    average_loss = forward_loss / num_targets.float()
            else:
                average_loss = torch.tensor(0.0).to(forward_targets.device)

            self._perplexity(average_loss)

            if num_targets > 0:
                return_dict.update(
                    {
                        "loss": average_loss,
                        "forward_loss": forward_loss / num_targets.float(),
                        "batch_weight": num_targets.float(),
                    }
                )
                if backward_loss is not None:
                    return_dict["backward_loss"] = backward_loss / num_targets.float()
            else:
                # average_loss zero tensor, return it for all
                return_dict.update({"loss": average_loss, "forward_loss": average_loss})
                if backward_loss is not None:
                    return_dict["backward_loss"] = average_loss

        return_dict.update(
            {
                # Note: These embeddings do not have dropout applied.
                "lm_embeddings": contextual_embeddings,
                "noncontextual_token_embeddings": embeddings,
                "mask": mask,
            }
        )

        return return_dict

    def get_metrics(self, reset: bool = False):
        return {"perplexity": self._perplexity.get_metric(reset=reset)}