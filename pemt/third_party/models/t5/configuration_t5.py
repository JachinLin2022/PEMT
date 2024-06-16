""" T5 model configuration """
from transformers.models.t5 import T5Config


class T5Config(T5Config):
    def __init__(self,
                 train_task_adapters=False,
                 prefix_tuning=False,
                 add_lora=False,
                 add_atten_lora=False,
                 lora_num=1,
                 atten_lora_rank=8,
                 source_task = None,
                 target_task = None,
                 add_task_embedding = None,
                 task_embedding_len = None,
                 task_embedding_init_token = None,
                 load_task_path = None,
                 init_task_from_vocab = None,
                 adapter_size = None,
                 num_experts = None,
                 inference_level = None,
                 sharing_down = None,
                 sharing_up = None,
                 apply_mixda = None,
                 num_of_kas = None,
                 adapter_down_scale = None,
                 layers=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.train_task_adapters = train_task_adapters
        self.prefix_tuning = prefix_tuning
        self.add_lora = add_lora
        self.lora_num = lora_num
        self.source_task = source_task
        self.target_task = target_task

        # for task embedding
        self.add_task_embedding = add_task_embedding
        self.task_embedding_len = task_embedding_len
        self.task_embedding_init_token = task_embedding_init_token
        self.load_task_path = load_task_path
        self.init_task_from_vocab = init_task_from_vocab
        
        # adamix
        self.adapter_size = adapter_size
        self.num_experts = num_experts
        self.inference_level = inference_level
        self.sharing_down = sharing_down
        self.sharing_up = sharing_up
        
        # mixda
        self.apply_mixda = apply_mixda
        self.num_of_kas = num_of_kas
        self.adapter_down_scale = adapter_down_scale
        self.layers = layers

        # origin_lora
        self.add_atten_lora = add_atten_lora
        self.atten_lora_rank = atten_lora_rank