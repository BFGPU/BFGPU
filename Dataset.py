from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer,max_length=1024,augment=None,output_idx=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.augment=augment
        self.output_idx=output_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text=self.texts[index]
        if self.augment:
            text=self.augment.augment(text)
        encoded_text = self.tokenizer(text, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt',return_token_type_ids=True)
        if self.output_idx:
            return index, encoded_text
        else:
            return encoded_text

class LabeledTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,max_length=1024,augment=None,output_idx=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.augment=augment
        self.output_idx=output_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text=self.texts[index]
        label = self.labels[index]
        if self.augment:
            text=self.augment.augment(text)
        encoded_text = self.tokenizer(text, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt',return_token_type_ids=True)
        if self.output_idx:
            return index, encoded_text, label
        else:
            return encoded_text, label