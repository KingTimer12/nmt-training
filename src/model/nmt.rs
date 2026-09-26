use burn::{
    Tensor,
    module::Module,
    nn::{Embedding, PositionalEncoding},
    tensor::{Bool, Float, Int, backend::Backend},
};

use crate::{
    model::{decoder::Decoder, encoder::Encoder},
    tokenizer::PAD,
};

#[derive(Module, Debug)]
pub struct NMT<B: Backend> {
    pub embedding: Embedding<B>,
    pub pos_encoding: PositionalEncoding<B>,
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
}

impl<B: Backend> NMT<B> {
    pub fn new(
        embedding: Embedding<B>,
        pos_encoding: PositionalEncoding<B>,
        encoder: Encoder<B>,
        decoder: Decoder<B>,
    ) -> Self {
        Self {
            embedding,
            pos_encoding,
            encoder,
            decoder,
        }
    }

    pub fn forward(
        &self,
        src: Tensor<B, 2, Int>,
        src_pad_mask: Tensor<B, 2, Bool>,
        tgt: Option<Tensor<B, 2, Int>>,
        tgt_pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3, Float> {
        let embeding_weight = self.embedding.weight.val();
        let src = self.embedding.forward(src);
        let src = self.pos_encoding.forward(src);
        let mut output = self.encoder.forward(src, src_pad_mask.clone());
        // println!("output shape: {:?}", output.dims());

        if let Some(tgt) = tgt
            && let Some(tgt_pad_mask) = tgt_pad_mask
        {
            let tgt = self.embedding.forward(tgt);
            let tgt = self.pos_encoding.forward(tgt);

            output = self
                .decoder
                .forward(tgt, tgt_pad_mask, output, src_pad_mask);
            // println!("output shape: {:?}", output.dims());
        }

        // Saída de notas, para saber a probabilidade de cada token do vocabulário ser o próximo token
        let [b, t, d] = output.dims();
        let output = output.reshape([b * t, d]);
        let output = output.matmul(embeding_weight.clone().transpose());
        output.reshape([b, t, embeding_weight.dims()[0]])
    }
}
