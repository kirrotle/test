# https://www.itread01.com/content/1548877326.html
# 為甚麼接property之後的def的名字都要長一樣呢

class data_item:

    def __init__(self):
        self.__difficult=0
        self.__text=""
        self.__bert_tranceform=0

    @property
    def difficult(self):
        return self.__difficult

    @difficult.setter
    def difficult(self,data):
        assert (data>=1 and data <=6),"difficult number wrong(1~6)"
        self.__difficult=data

    @property
    def composition(self):
        return self.__text

    @composition.setter
    def composition(self,data):
        assert isinstance(data,str),"data must be sting format"
        self.__text=data

    @property
    def bert_tranceform(self):
        return self.__bert_tranceform

    @ bert_tranceform.setter
    def bert_tranceform(self,data):
        self.__bert_tranceform=data
        

    #property也可以這樣寫
    #difficult=property(get_difficult,set_difficult)

#test
if __name__=="__main__":
    x=data_item()
    x.difficult=6
    x.composition="哈囉"
    print(x.difficult)
    print(x.composition)



