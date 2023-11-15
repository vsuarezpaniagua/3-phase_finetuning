from sentence_transformers import SentenceTransformer, models
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn


class Classifier_head(nn.Module):
    """
    This class is the classifier head model.
    It can be tuned to use one or two different extra layers with different dropouts 
    """

    def __init__(self,
                 config,
                 num_class=0,
                 drop_1=0.3,
                 drop_2=0.3,
                 architecture="",
                 middle_layer=100):
        """
        This class if the neural classifier on the top of the pretrained model.
        :param config: BaseConfig object with the configuration for the encoder
        :param num_class: int, number of categories if it is a classification problem
        :param drop_1: float, first dropout for the complex architecture
        :param drop_2: float, second dropout for the complex architecture
        :param architecture: str, "simple" or "complex"
        :param middle_layer: int, number of units of the middle layer
        """
        super(Classifier_head, self).__init__()

        # base model and architecture
        self.architecture = architecture
        self.num_class = num_class
        self.config = config
        num_class = 1 if "reg" in self.config.model_type else num_class

        # Base sentence model
        self.encoder = EncoderBlock(config)

        # Losses
        if "class" in config.model_type:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

        # First dropout
        self.drop1 = nn.Dropout(drop_1)

        # Architecture for extraction
        if self.architecture == "simple":
            self.class_output = nn.Linear(self.config.embedding_size, self.num_class)

        if self.architecture == "complex":
            self.class_mid = nn.Linear(self.config.embedding_size, middle_layer)
            self.drop2 = nn.Dropout(drop_2)
            self.class_output = nn.Linear(middle_layer, self.num_class)

    def forward(self, sample, class1_target):
        """
        input:
            sample: batch of sentences, shape = (batch_size, embedding_size)
            class_target: labels dimensions = (batch_size,)
        """
        embedding_batch = self.encoder(sample)
        embedding_batch = embedding_batch.to(self.config.device)
        class_logits = None
        loss = 0

        # Simple architecture
        if self.architecture == "simple":
            output_class_1 = self.drop1(embedding_batch)
            # We add the linear outputs
            class_logits = self.class_out(output_class_1)

        # Complex architecture
        if self.architecture == "complex":
            # Add dropout 1
            output_class_1 = self.drop1(embedding_batch)
            # Add middle layer
            output_class_2 = self.class_mid(output_class_1)
            # Add second dropout
            output_class_3 = self.drop2(output_class_2)
            # We add the linear outputs
            class_logits = self.class_output(output_class_3)

        if "class" in self.config.model_type:
            loss = self.loss(class_logits.view(-1, self.num_class),
                             class1_target.view(-1))
        if "reg" in self.config.model_type:
            loss = self.loss(class_logits, class1_target)

        return class_logits, loss


class BaseConfig:
    def __init__(self,
                 base_model_path,
                 device,
                 freezing=False,
                 embedding_size=768,
                 transformer=True,
                 model_type="classification",
                 truncation=512,
                 from_embeddings=False,
                 **kwargs
                 ):
        """
        :param base_model_path: str/path, path to load the model from
        :param freezing: Bool, if we may free the neurons on the embedder of not
        :param device: cpu or gpu
        :param embedding_size: int, size of the embedding, 768 for BERT for example
        :param transformer: boolean, True if it is a transformer, False if it is a sentence transformer
        :param model_type:
        :param from_embeddings:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.freezing = freezing
        self.device = device
        self.base_model_path = base_model_path
        self.embedding_size = embedding_size
        self.transformer = transformer
        self.model_type = model_type
        self.truncation = truncation
        self.from_embeddings = from_embeddings


class EncoderBlock(nn.Module):
    """  This class an encoder head for the model
    config must be an encoder config instance """

    def __init__(self,
                 config
                 ):
        super(EncoderBlock, self).__init__()
        self.config = config
        print("Configuration EncoderBlock")
        for x, y in config.__dict__.items():
            if "_" not in x:
                print(x,y)

        #  Construction of the encoder: transformers, sentence transformer, and no encoder
        if self.config.transformer:
            config = AutoConfig.from_pretrained(self.config.base_model_path)
            self.model = AutoModel.from_config(config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            word_embedding_model = models.Transformer(self.config.base_model_path)
            self.output_size = word_embedding_model.get_word_embedding_dimension()
        elif not self.config.from_embeddings:
            self.model = SentenceTransformer(self.config.base_model_path, device=self.config.device)
            # self.output_size = self.model.get_word_embedding_dimension()
        else:
            self.model = False
            self.output_size = self.config.embedding_size

        #   Freezing or not the encoder
        if self.config.freezing and self.model:
            for param in self.model.parameters():
                param.requires_grad = False
        elif self.model:
            for param in self.model.parameters():
                param.requires_grad = True

        self.model.to(self.config.device)

    def forward(self, sample=None):

        if self.config.transformer:
            inputs = self.tokenizer(sample, truncation=True, return_tensors="pt",
                                    max_length=self.config.truncation, padding=True)
            inputs = inputs.to(self.config.device)
            output = self.model(**inputs)
            embedding_batch = output["pooler_output"]

        elif not self.config.from_embeddings:
            embedding_batch = self.model.encode(sentences=sample,
                                                convert_to_numpy=False,
                                                convert_to_tensor=True,
                                                device=self.config.device)
        else:
            embedding_batch = sample
        return embedding_batch
