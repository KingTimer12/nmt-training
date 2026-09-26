use burn::{
    Tensor,
    module::Module,
    nn::{
        Embedding, LayerNorm, PositionalEncoding,
        transformer::{TransformerDecoder, TransformerDecoderInput},
    },
    tensor::{Bool, Float, Int, backend::Backend},
};

use crate::tokenizer::PAD;

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
        //todos os elementos do tensor que são iguais ao PAD serão true, e os outros false.
        // let tgt_mask = tgt.equal_elem(PAD);
        // let memory_mask = src.equal_elem(PAD);
        let r = TransformerDecoderInput::new(tgt, memory)
            .target_mask_pad(tgt_mask)
            .memory_mask_pad(memory_mask);
        let r = self.transformer_decoder.forward(r);
        self.layer_norm.forward(r)
    }
}
