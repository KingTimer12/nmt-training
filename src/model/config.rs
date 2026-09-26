use burn::{
    config::Config,
    nn::{
        EmbeddingConfig, LayerNormConfig, PositionalEncodingConfig,
        transformer::{TransformerDecoderConfig, TransformerEncoderConfig},
    },
    tensor::backend::Backend,
};

use crate::model::{decoder::Decoder, encoder::Encoder, nmt::NMT};

#[derive(Config, Debug)]
pub struct NMTConfig {
    pub n_embedding: usize,
    pub d_model: usize,
}

impl NMTConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NMT<B> {
        let embeding = EmbeddingConfig::new(self.n_embedding, self.d_model).init(device);
        let pos_enc = PositionalEncodingConfig::new(self.d_model).init(device);

        let encoder = EncoderConfig::new(self.n_embedding, self.d_model).init(device);
        let decoder = DecoderConfig::new(self.n_embedding, self.d_model).init(device);
        NMT::new(embeding, pos_enc, encoder, decoder)
    }
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub n_embedding: usize,
    pub d_model: usize,
}

impl EncoderConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B> {
        // d_model: o mesmo 256 de antes. Tem que ser igual ao do embedding.
        // n_heads: quantas "atenções" rodam em paralelo.
        // Cada uma pode prestar atenção em um tipo de relação diferente (uma no sujeito, outra no verbo, etc.).
        // Use 4. O d_model precisa ser divisível por esse número.
        // d_ff: depois da atenção, cada token passa por uma pequena camada de processamento própria, e esse é o tamanho interno dela.
        // O costume é 4 × d_model, então 1024.
        // n_layers: quantas vezes esse processo (atenção + processamento) se repete,
        // uma camada em cima da outra. Comece com 3.
        // dropout: durante o treino, desliga aleatoriamente uma parte dos números para o modelo não "decorar" os exemplos.
        // Use 0.1. (já é default)
        // norm_first: coloque true. É uma variação que deixa o treino mais estável.
        let transformer_encoder =
            TransformerEncoderConfig::new(self.d_model, self.d_model * 4, 4, 3)
                .with_norm_first(true)
                .init(device);

        let layer_norm = LayerNormConfig::new(self.d_model).init(device);
        Encoder::new(transformer_encoder, layer_norm)
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    pub n_embedding: usize,
    pub d_model: usize,
}

impl DecoderConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        // d_model: o mesmo 256 de antes. Tem que ser igual ao do embedding.
        // n_heads: quantas "atenções" rodam em paralelo.
        // Cada uma pode prestar atenção em um tipo de relação diferente (uma no sujeito, outra no verbo, etc.).
        // Use 4. O d_model precisa ser divisível por esse número.
        // d_ff: depois da atenção, cada token passa por uma pequena camada de processamento própria, e esse é o tamanho interno dela.
        // O costume é 4 × d_model, então 1024.
        // n_layers: quantas vezes esse processo (atenção + processamento) se repete,
        // uma camada em cima da outra. Comece com 3.
        // dropout: durante o treino, desliga aleatoriamente uma parte dos números para o modelo não "decorar" os exemplos.
        // Use 0.1. (já é default)
        // norm_first: coloque true. É uma variação que deixa o treino mais estável.
        // let transformer_encoder =
        //     TransformerEncoderConfig::new(self.d_model, self.d_model * 4, 4, 3)
        //         .with_norm_first(true)
        //         .init(device);
        let transformer_decoder =
            TransformerDecoderConfig::new(self.d_model, self.d_model * 4, 4, 3)
                .with_norm_first(true)
                .init(device);

        let layer_norm = LayerNormConfig::new(self.d_model).init(device);
        Decoder::new(transformer_decoder, layer_norm)
    }
}
