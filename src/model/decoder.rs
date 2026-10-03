use burn::{
    Tensor, module::Module, nn::{
        LayerNorm, attention::generate_autoregressive_mask, transformer::{TransformerDecoder, TransformerDecoderInput},
    }, tensor::{Bool, Float, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    transformer_decoder: TransformerDecoder<B>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(transformer_decoder: TransformerDecoder<B>, layer_norm: LayerNorm<B>) -> Self {
        Self {
            transformer_decoder,
            layer_norm,
        }
    }

    pub fn forward(
        &self,
        tgt: Tensor<B, 3, Float>,
        tgt_mask: Tensor<B, 2, Bool>,
        memory: Tensor<B, 3, Float>,
        memory_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3, Float> {
        let [b, t, _] = tgt.dims();
        let causal = generate_autoregressive_mask(b, t, &tgt.device());
        let r = TransformerDecoderInput::new(tgt, memory)
            .target_mask_attn(causal)
            .target_mask_pad(tgt_mask)
            .memory_mask_pad(memory_mask);
        let r = self.transformer_decoder.forward(r);
        self.layer_norm.forward(r)
    }
}
