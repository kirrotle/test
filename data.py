import numpy as np
from data_item import data_item
import gluonnlp
import mxnet

class data:
    def __init__(self):
        self.model,self.bert_sentence_transform=self.load_bert_model()
        self.data=self.read_data()
        print("data ok")

    def load_bert_model(self):
        model,vocab=gluonnlp.model.get_model("bert_12_768_12",dataset_name='wiki_cn_cased',use_classifier=False, use_decoder=False)
        tokenizer = gluonnlp.data.BERTTokenizer(vocab, lower=True)
        transform = gluonnlp.data.BERTSentenceTransform(tokenizer, max_seq_length=128, pair=False, pad=False)
        return model,transform

    def read_data(self):
        name=self.read_file("./0991/record.txt","UTF-16")
        name=np.reshape(np.array(name),(6,200))
        name_dic={}
        for difficult in range(1,7):
            for file in name[difficult-1]:
                tmp=data_item()
                text_array=self.read_file("./0991/{}/{}.txt".format(str(difficult),file),"cp950")
                
                #檔案中有**##要去掉
                for text in text_array:
                    tmp.composition=tmp.composition+text.replace("**##","")

                tmp.difficult=difficult
                
                transform=self.bert_sentence_transform([tmp.composition])
                words, valid_len, segments = mxnet.nd.array([transform[0]]), mxnet.nd.array([transform[1]]), mxnet.nd.array([transform[2]])
                seq_encoding,cls_encoding = self.model(words, segments, valid_len)
                tmp.bert_tranceform=cls_encoding.asnumpy().tolist()[0]
                name_dic.update({file:tmp})
                print(file)
        return name_dic

    
    def read_file(self,file_name,file_encoding):
        with open(file_name,mode="r",encoding=file_encoding) as file:
            lines=file.readlines()
            lines=[i.replace("\n","") for i in lines]
            return lines
        

#test
if __name__=="__main__":
    x=data().data
    print(x["0001"].composition.encode("UTF-8"))
    