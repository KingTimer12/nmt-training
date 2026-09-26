use burn::{
    Tensor,
    module::Module,
    nn::{
        LayerNorm,
        transformer::{TransformerEncoder, TransformerEncoderInput},
    },
    tensor::{Bool, Float, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    transformer_encoder: TransformerEncoder<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(transformer_encoder: TransformerEncoder<B>, layer_norm: LayerNorm<B>) -> Self {
        Self {
            transformer_encoder,
            layer_norm,
        }
    }

    pub fn forward(
        &self,
        src: Tensor<B, 3, Float>,
        pad_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3, Float> {
        //todos os elementos do tensor que são iguais ao PAD serão true, e os outros false.
        let r = TransformerEncoderInput::new(src).mask_pad(pad_mask);
        let r = self.transformer_encoder.forward(r);
        self.layer_norm.forward(r)
    }
}
