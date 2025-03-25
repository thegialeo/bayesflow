from collections.abc import Sequence

import keras
from keras import ops, layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from .invariant_module import InvariantModule


@serializable(package="bayesflow.networks")
class EquivariantModule(keras.Layer):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        mlp_widths_equivariant: Sequence[int] = (128, 128),
        mlp_widths_invariant_inner: Sequence[int] = (128, 128),
        mlp_widths_invariant_outer: Sequence[int] = (128, 128),
        pooling: str = "mean",
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: int | float | None = 0.05,
        layer_norm: bool = True,
        spectral_normalization: bool = False,
    ):
        """
        Initializes an equivariant module that combines equivariant transformations with nested invariant transforms
        to enable interactions between set members.

        This module applies an equivariant transformation to each set member, followed by an invariant transformation
        that aggregates information and injects it back into the set representation.

        The architecture consists of a fully connected residual block for equivariant processing and an invariant
        module to enhance expressiveness.

        The model supports different activation functions, dropout, layer normalization, and optional spectral
        normalization for stability.

        Parameters
        ----------
        mlp_widths_equivariant : Sequence[int], optional
            Widths of the MLP layers in the equivariant transformation applied to each set member.
            Default is (128, 128).
        mlp_widths_invariant_inner : Sequence[int], optional
            Widths of the inner MLP layers in the invariant module. Default is (128, 128).
        mlp_widths_invariant_outer : Sequence[int], optional
            Widths of the outer MLP layers in the invariant module. Default is (128, 128).
        pooling : str, optional
            Type of pooling operation used in the invariant module, such as "mean". Default is "mean".
        activation : str, optional
            Activation function applied in the MLP layers, such as "gelu". Default is "gelu".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        dropout : int, float, or None, optional
            Dropout rate applied within the MLP layers. Default is 0.05.
        layer_norm : bool, optional
            Whether to apply layer normalization after transformations. Default is True.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        """

        super().__init__()

        # Invariant module to increase expressiveness by concatenating outputs to each set member
        self.invariant_module = InvariantModule(
            mlp_widths_inner=mlp_widths_invariant_inner,
            mlp_widths_outer=mlp_widths_invariant_outer,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=pooling,
            spectral_normalization=spectral_normalization,
        )

        # Fully connected net + residual connection for an equivariant transform applied to each set member
        self.input_projector = layers.Dense(mlp_widths_equivariant[-1])
        self.equivariant_fc = keras.Sequential()
        for width in mlp_widths_equivariant:
            layer = layers.Dense(
                units=width,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.equivariant_fc.add(layer)

        self.layer_norm = layers.LayerNormalization() if layer_norm else None

    def build(self, input_shape):
        self.input_projector.build(input_shape)
        input_shape = self.input_projector.compute_output_shape(input_shape)

        self.invariant_module.build(input_shape)
        summary_shape = self.invariant_module.compute_output_shape(input_shape)

        input_shape = *input_shape[:-1], input_shape[-1] + summary_shape[-1]

        self.equivariant_fc.build(input_shape)
        input_shape = self.equivariant_fc.compute_output_shape(input_shape)

        if self.layer_norm is not None:
            self.layer_norm.build(input_shape)

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        input_set : Tensor
            The input tensor representing a set, with shape
            (batch_size, ..., set_size, input_dim).
        training : bool, optional
            A flag indicating whether the model is in training mode. Default is False.
        **kwargs : dict
            Additional keyword arguments for compatibility with other functions.

        Returns
        -------
        Tensor
            The transformed output tensor with the same shape as `input_set`, where
            each element is processed through the equivariant transformation.
        """

        input_set = self.input_projector(input_set)

        # Example: Output dim is (batch_size, ..., set_size, representation_dim)
        invariant_summary = self.invariant_module(input_set, training=training)

        invariant_summary = keras.ops.expand_dims(invariant_summary, axis=-2)
        invariant_summary = keras.ops.broadcast_to(invariant_summary, keras.ops.shape(input_set))

        # Concatenate each input entry with the repeated invariant embedding
        output_set = ops.concatenate([input_set, invariant_summary], axis=-1)

        # Pass through final equivariant transform + residual
        output_set = input_set + self.equivariant_fc(output_set, training=training)
        if self.layer_norm is not None:
            output_set = self.layer_norm(output_set, training=training)

        return output_set

    def compute_output_shape(self, input_shape):
        output_shape = self.input_projector.compute_output_shape(input_shape)
        summary_shape = self.invariant_module.compute_output_shape(output_shape)

        output_shape = *output_shape[:-1], output_shape[-1] + summary_shape[-1]

        output_shape = self.equivariant_fc.compute_output_shape(output_shape)

        if self.layer_norm is not None:
            output_shape = self.layer_norm.compute_output_shape(output_shape)

        return output_shape
