import urllib.parse as url 
import numpy as np
import pandas as pd
import random
import linecache
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import FastText  # another type of fasttext
import numpy as np

def Delete_repeated_data(file_in='Split-normal.csv',file_out='unique.csv'):
    '''
    delete the repeated data ina file
    '''
    file0=open(file_in,'r',errors='replace')
    file1=open(file_out,'wt',errors='replace')
    sets=[]
    for line in file0:
        if line not in sets:
            sets.append(line)
            file1.writelines(line)

def Extratc_sqli(file_in,file_out):
    '''
    extract sqli from source data
    '''
    file1=open(file_in,'r')
    file2=open(file_out,'wt')
    for line in file1:
        if 'sqli' in line:
            line=line.split(',"sqli",')
            line2=line[0].split(',"')
            line2=line2[0][1:-1]
            file2.writelines(line2+'\n')
    # file_in = 'Source/payload_train.csv'
    # file_out= 'Extract-Source/data-sql.csv'
    # Extratc_sqli(file_in,file_out)

def Extratc_normal(file_in,file_out):
    '''
    extract normal from source data
    '''
    file1=open(file_in,'r')
    file2=open(file_out,'wt')
    for line in file1:
        if '"norm"' in line:
            line=line.split(',"norm",')
            line2=line[0].split(',"')
            line2=line2[0][1:-1]
            file2.writelines(line2+'\n')
# file_in = 'Source/payload_test.csv'
# file_out= 'Extract-Source/new-test.csv'
# Extratc_normal(file_in,file_out)

def CreateDict(keyworsd):
    '''
    Create dcts for key words
     keywords: a long string as 'word1 word2 word3 ... ' 
     return: a dict with all words from keywords
    '''
    dict_ReservedWords={}
    words=keyworsd.split(' ')
    for num in range(len(words)):
        if words[num] not in dict_ReservedWords:
            dict_ReservedWords[words[num]]=0
    return dict_ReservedWords

def filterline_new(files_in,files_out):
    '''
    This function is to extract/split words from sentences from the 
    files_in and save them in files_out 
     files_in : the path of source files
     files_out : the path to save the extracted sentences 
    '''
    file_in=open(files_in,'r',errors='ignore')
    file_out=open(files_out,'wt',errors='ignore')
    for line in file_in:
        # # List to string
        # line="".join(line)
        
        # --------- Filtering url code with decode all %xx 
        line1=''
        while '%' in line and line1!=line:
            line1=line
            letter=[]
            for j in range (len(line)):
                # print (j)
                if line[j]=='%':
                    letter.append(line[j:j+3])
                    # print (line[j])
            for i in range (len(letter)):
                line=line.replace(letter[i],url.unquote(letter[i])) # the string will not be changed if the string is not a url-encoded.
        # ----------------- end -----------------

        line=line.lower()
        line=line.replace('%',' % ')
        line=line.replace('/',' / ')
        line=line.replace('+',' + ')
        line=line.replace('?',' ? ')
        line=line.replace('&',' & ')
        line=line.replace(';',' ; ')
        line=line.replace('=',' = ')
        line=line.replace(',',' , ')
        line=line.replace('\'',' \' ')
        line=line.replace('\"',' \" ')
        # add
        line=line.replace('.',' . ')
        line=line.replace('(',' ( ')
        line=line.replace(')',' ) ')
        line=line.replace('<',' < ')
        line=line.replace('>',' > ')
        line=line.replace('*',' * ')
        line=line.replace('!',' ! ')
        line=line.replace('#',' # ')
        line=line.replace('|',' | ')
        line=line.replace('$',' $ ')
        line=line.replace('^',' ^ ')
        line=line.replace('{',' { ')
        line=line.replace('}',' } ')
        line=line.replace('--',' -- ')
        line=line.replace('@',' @ ')
        line=line.replace('\\',' \\ ')

        # Enter 
        line=line.replace('\r',' ')
        line=line.replace('\n',' ')
        # line=line[:-1]


        # print (line)
        # file_out.writelines(new_line+'\n')
        # line=line.decode('unicode-escape')
        file_out.writelines(line+'\n')

def is_Pure_strings(string):
    '''
    check a string is PureString or mixed string
    '''
    number=['0','1','2','3','4','5','6','7','8','9']
    
    for i in range(len(number)):
        if number[i] in string:
            result= False
            break
        else: 
            result= True 
    return result


def Replace_words(files_in,files_out,key_words):
    '''
    Repalce words those are not in keywords and Punctuations below.
     Arg:
      files_in: the file of the data 
      files_out: the file to save those data replaced
    '''
    # words=Find_keywords()
    Punctuations=['/','+','?','&',';','=',',','\'','\"','(',')','<','>','*','!','$','#','|','^','{','}','\\','--','%','~','@','.','`','[',']',':'] 
    # key_words=Find_keywords()
    # print (key_words)
    all_keywords=Punctuations+key_words
    file0=open(files_in,'r',errors='ignore')
    file1=open(files_out,'wt')
    
    for line in file0:
        New_line=[]
        line=line.strip()
        line=line.split(' ')
        for i in range(len(line)):
            if line[i] in (Punctuations+all_keywords):
                New_line.append(line[i])
            if line[i] not in all_keywords and line[i] !='' :
                if line[i].isdigit():
                    New_line.append('Numbers')
                elif is_Pure_strings(line[i]):
                    New_line.append('PureString')
                else:
                    New_line.append('MixString')
        newline=' '.join(New_line)
        file1.writelines(newline+'\n')        
# ------------- 

# the new one for unkonown words in test 
def Find_keywords_New(file_in,file_keywords):
    '''
    find small set of key words from statistics
    Counts the frequency of all words below and extract the top num of words 
    ----
     file_in: file we will extract key words from.  
     num: the num of words will be extracted (changable, defaults=80)
     return: 
       smallkey_words: the top xx key words appeared in training data
       allkey_words: all key words 
    '''

    Html_word='doctype a abbr acronym address applet area article aside audio b base basefont bdi bdo big blockquote body br button canvas caption center cite code col colgroup command datalist dd del details dfn dialog dir div dl dt em embed fieldset figcaption figure font footer form frame frameset h1 h2 h3 h4 h5 h6 head header hr html i iframe img input ins kbd keygen label legend li link main map mark menu menuitem meta nav noframes no script object ol optgroup option output p param pre progress q rp rt ruby s samp' # delete ''
    Script_word='section select small source span strike strong style sub summary sup table tbody td textarea tfoot th thead time title tr track tt u ul var video wbr script javascript'
    Script_function='click close alert confirm escape eval isnan parsefloat parseint prompt unescape join length reverse sort getdate getday gethouse getminutes getmonth getsecond gettime gettimezoneoffset getyear parse setdate sethours setminutes setmonth settime setyear togmtstring setlocalstring utc math e ln2 ln10 log2e log10e pi sqr1_2 sqrt2 abs acos asin atan atan2 ceil cos exp floor log max min pow random round sin sqrt tan anchor big blink bold charat fixed fontcolor fontsize indexof italics lastindexof length link small strike sub substring sup tolowercase touppercase trim document write open writln'
    # sql words 
    Reserved_word='select update delete insert create alter drop order by group by truncate replace commit rollback savepoint transaction set distinct all desc null limit top percent rownum as having inner left right full outer self index table tables databases database column auto_increment view default unique check constraint key primary foreign modify information_schema false true where if union between like in and or not into is join adddate addtime convert_tz else from'

    function_time='adddate addtime convert_tz current_date curdate current_time current_timestamp curtime date_add date_format date_sub date datediff day dayname dayofmonth dayofweek dayofyear extract from_days from_unixtime hour last_day localtime localtimestamp makedate maketime microsecond minute month monthname now period_add period_diff quarter sec_to_time second str_to_date subdate subtime sysdate time_format time_to_sec time timediff timestamp timestampadd timestampdiff to_days unix_timestamp utc_date utc_time utc_timestamp week weekday weekofyear year yearweek'

    function_other='max min count avg sum first last ucase lcase mid len round format field upper lower sqrt rand concat isnull nvl ifnull replace trim version user dataset substring elt group_concat concat_ws extractvalue updatexml sleep make_set benchmark extractvalue cast'

    Orcal_words='exp power mod ceil floor sign aprt ascii chr instr length substr ltrim rtrim soundex initcap add_months mouths_between current_date to_char to_date to_blob to_clob to_number decode coalesce trunc next_day lpad rpad nlssort abs greatest vsize regexp_substr regexp_instr repeat'

    # ------------------ words for other attacks
    others='data text plain php input phpinfo proc self cmdline stat status fd fckeditor access_log access log error_log error stats_log license_log login_log mysql bin slow pure ftpd purftpd mainlog paniclog rejectlog mailog conf bin logs sbin base64_decode utility convert sysobject java lang wget curl redirect shal regexp master ping exec system exe'
    
    add='shell_exec passthru pcntl_exec popen proc_open echo src cookie case'

    all_words =  Html_word+' '+Script_word+' '+Script_function+' '+Reserved_word + ' ' + function_time + ' ' + function_other + ' ' + Orcal_words+' '+others+' '+add
    all_dicts = CreateDict(all_words)
    # dict_out='./keywords.csv'
    file=open(file_in,'r', errors='ignore')
    for line in file:
        line=line.strip()
        line=line.split(' ')
        for i in range(len(line)):
            if line[i]!='' and line[i] in all_dicts:
                all_dicts[line[i]] += 1

    # sorted the dicts according their frequency
    sorted_dicts = sorted(all_dicts.items(),key=lambda item:item[1],reverse = True)
    all_num=len(sorted_dicts)
    # extract the frequency counts for all words
    sorted_freq= [word[1] for word in sorted_dicts[0:all_num]]
    num=0
    # find the last word with frequency >= 1
    for i in range(all_num):
        if sorted_freq[i]==0:
            num=i
            # print('sorted: ',num)
            # print('sorted data: ', sorted_dicts[:num])
            break
    print('key words number:', num-20)
    # --------- save the dicts 
    # file2 = open(dict_out,'wt')
    # for pairs in sorted_dicts:
    #     line=str(pairs)
    #     file2.writelines(line+'\n')
    # --------------- extract xx words from the sorted dicts 
    small_dicts = [word[0] for word in sorted_dicts[0:num-20]] # delete the space
    allkey_words=all_words.split(' ')
    smallkey_words=" ".join(small_dicts)
    file_save=open(file_keywords,'wt')
    file_save.writelines(smallkey_words)
    if '' in small_dicts:
        print (1)
    # print (small_dicts)
    return small_dicts,allkey_words

def Replace_words_New(files_in,files_out,samllkey_words,allkey_words):
    '''
    Repalce words those are not in keywords and Punctuations below.
     Arg:
      files_in: the file of the data 
      files_out: the file to save those data replaced
    '''
    # words=Find_keywords()
    Punctuations=['/','+','?','&',';','=',',','\'','\"','(',')','<','>','*','!','$','#','|','^','{','}','\\','--','%','~','@','.','`','[',']',':']
    # key_words=Find_keywords('change')
    # print (key_words)
    all_keywords=Punctuations+allkey_words
    file0=open(files_in,'r',errors='ignore')
    file1=open(files_out,'wt')
    
    for line in file0:
        New_line=[]
        line=line.strip()
        line=line.split(' ')
        for i in range(len(line)):
            if line[i] =='':
                continue
            if line[i] in (Punctuations+samllkey_words):
                New_line.append(line[i])
            elif line[i] in all_keywords :
                New_line.append('Sen_words')
            else:
                if line[i].isdigit():
                    New_line.append('Numbers')
                elif is_Pure_strings(line[i]):
                    New_line.append('PureString')
                else:
                    New_line.append('MixString')

            # elif line[i] not in (samllkey_words+Punctuations) and line[i] !=' ' :
                # print(1)
                # if line[i].isdigit():
                #     line[i]='Sen_Numbers'
                # if is_Pure_strings(line[i]):
                #     line[i]='Sen_PureString'
                # else:
                #     line[i]='Sen_MixString'
                
        newline=' '.join(New_line)
        file1.writelines(newline+'\n')        
# --------------


def Random_Num(index,number):
    '''
    Generate a list of index from index randomly
     index : the index of files
     number : the length of list generated
     return : a list of index  
    '''
    return random.sample(list(index),number)

def Disorder(file_in,file_out,total_num):
    '''
    disorder the data 
     Arg:
      total_num: the number of the data
    '''
    index=[n for n in range(total_num)]
    index_random=Random_Num(index,total_num)
    # file0=open(file_in,'r')
    file1=open(file_out,'wt',encoding='utf-8',errors='replace')
    for i in range(total_num):
        j=index_random[i]
        line=linecache.getline(file_in,j+1)
        file1.writelines(line)

    # file_in='Extract-Source/Replace-normal.csv'
    # file_out='Extract-Source/Replace-disorder-normal.csv'
    # total_num=10375
    # Disorder(file_in,file_out,total_num)

def Data_mix(file1_in,file2_in,total_number,normal_number,file_out,label_out):
    '''
    Mix two files (normal and anomalous) into one
    ------
     Arg:
      file1_in:  the normal data (label: 0)
      file2_in:  the anomalous data (label: 1)
      total_number: the total number of the two files
      normal_number: the number of the normal data
      file_out: the mixed data 
      label_out: the mixed label

    '''
    file0=open(file_out,'wt',encoding='utf-8',errors='replace')
    # file1=open(label_out,'wt',encoding='utf-8',errors='replace')
    index=[n for n in range(total_number)]
    index_randm=Random_Num(index,normal_number)
    index_randm=sorted(index_randm)
    index_randm.append(0)
    print (len(index_randm))
    j=0
    k=0
    data_label=[]
    for i in range(total_number):
        if i==index_randm[j]:
            line=linecache.getline(file1_in,j+1)
            file0.writelines(line)
            j=j+1
            # file1.writelines('0,1\n')
            data_label.append(0)
        else:
            # print(i)
            line=linecache.getline(file2_in,k+1)
            file0.writelines(line)
            k=k+1
            # file1.writelines('1,0\n')
            data_label.append(1)
    print('train number:',j)
    print('test number:',k)
    np.savetxt(label_out,data_label,fmt='%d')


def Extract_Train_Test(file_in,label_in,file_train,labe_train,file_test,label_test,number):
    '''
    Extract train and test data from mixed data.
     Arg: 
       file_in: the mixed data
       label_in: the mixed labels
       file_train: the train data
       label_train: the train labels 
       file_test: the test data
       label_test: the test labels
       number: the number of train data
    '''
    file0=open(file_in,'r',encoding='utf-8',errors='replace')
    label0=open(label_in,'r',encoding='utf-8',errors='replace')
    file1=open(file_train,'wt',encoding='utf-8',errors='replace')
    label1=open(labe_train,'wt',encoding='utf-8',errors='replace')
    fiel2=open(file_test,'wt',encoding='utf-8',errors='replace')
    label2=open(label_test,'wt',encoding='utf-8',errors='replace')
    num_data=0
    num_label=0
    for line in file0:
        if num_data< number:
            file1.writelines(line)
        else:
            fiel2.writelines(line)
        num_data+=1

    for label in label0:
        if num_label< number:
            label1.writelines(label)
        else:
            label2.writelines(label)
        num_label+=1

def Word_2vec_Save(files_in,models_out,min_count,size,iters):
    '''
    Word2vec Model : input sentences in which words are pulled together with  ' ' 
     The input sentence is a list in a [] --> [list,list2,list3,......] and save the model.
    Arg:
     files_in: the tokenized txt data
     models_out: the path to save the model
     min_count: the threshold of frequency, if a word's frequency is smaller than it, the word will be drop
     size: the size of output vector of words
     iters: the times for training
    '''
    txt_file=open(files_in,'r')
    sentences=[]
    for line in txt_file:
        # print (line)
        line=line.strip()
        line=line.split(' ')
        sentences.append(line)
    # Train the networks
    model=gensim.models.Word2Vec(sentences,min_count=min_count,size=size,iter=iters,window=7)
    model.save(models_out)

def FastText_Save(files_in,models_out,min_count,size,iters):
    '''
    FastText Model : input sentences in which words are pulled together with  ' ' 
     The input sentence is a list in a [] --> [list,list2,list3,......] and save the model.
    Arg:
     files_in: the tokenized txt data
     models_out: the path to save the model
     min_count: the threshold of frequency, if a word's frequency is smaller than it, the word will be drop
     size: the size of output vector of words
     iters: the times for training
    // vocabulary: vocab=(model.wv.vocab).keys()
    '''
    txt_file=open(files_in,'r')
    sentence=[]
    for line in txt_file:
        line=line.strip()
        line=line.split(' ')
        sentence.append(line)
    # Train the networks
    model=FastText(sentence,min_count=min_count,size=size,iter=iters,window=5)
    model.save(models_out)


def count_max_min(file_name,max_len=0,min_len=50):
    files_in=open(file_name,'r')
    for line in files_in :
        line=line.split(' ')
        length=len(line)
        if length > max_len:
            max_len=length
        if length < min_len:
            min_len=length
    print('max: ',max_len,'min: ',min_len )

def count_stage(file_name,threshold):
    num_lower=0
    num_higher=0
    files_in=open(file_name,'r')
    for line in files_in :
        line=line.split(' ')
        length=len(line)
        if length >=threshold:
            num_higher+=1
        else:
            num_lower+=1
    print('bigger: ',num_higher,'lower: ',num_lower )

def encode(model_file,type_model,source_file,files_out,thres_num=80,vec_length=48):
    '''
    Encoding the splitted data into a matrix with shape of 80*48 by word2vec algorithm
    the num
     Arg:
       model_file: the path of the model of word2vec or FastText 
       source_file: the path of the data which is read to be encoded 
       files_out: the path to save the encoded data
       type_model: 1——Word2vec  others: FastText
    '''
    if type_model==1:
        model=gensim.models.Word2Vec.load(model_file)
    else:
        model=FastText.load(model_file)
    file_in=open(source_file,'r')
    file_out=open(files_out,'wt')
    zeros=np.zeros([vec_length])
    for line in file_in:
        # print(line)
        line=line.strip()
        data_senten=[]
        line=line.split(' ')
        if '' in line:
            print (line)
        length=len(line)
        if length < thres_num:
            for i in range(length):
                if line[i] not in model.wv.vocab.keys():
                    relaced_data='MixString'
                    data_senten=np.append(data_senten,model[relaced_data])
                else:
                    data_senten=np.append(data_senten,model[line[i]])
            for i in range (thres_num-length):
                data_senten=np.append(data_senten,zeros)
        if length >= thres_num:
            for i in range (thres_num):
                if line[i] not in model.wv.vocab.keys():
                    relaced_data='MixString'
                    data_senten=np.append(data_senten,model[relaced_data])
                else:
                    data_senten=np.append(data_senten,model[line[i]])
        data_senten=np.reshape(data_senten,(-1,thres_num*vec_length))
        # data_senten
        # print(np.shape(data_senten))
        np.savetxt(file_out,data_senten,delimiter=',',fmt='%f')

# this function is for three classes experiment.
def Data_mix_sqli_norm_xss(file1_in,file2_in,label1_in,label2_in,total_number,xss_number,file_out,label_out):
    '''
    Mix two files (xss and sql+normal ) into one
    ------
     Arg:
      file1_in:  the xss data
      file2_in:  the sql+normal data
      total_number: the total number of the two files
      xss_number: the number of the xss data
      file_out: the mixed data 
      label_out: the mixed label

    '''
    file0=open(file_out,'wt',errors='replace')
    file1=open(label_out,'wt')
    index=[n for n in range(total_number)]
    index_randm=Random_Num(index,xss_number)
    index_randm=sorted(index_randm)
    index_randm.append(0)
    print (len(index_randm))
    j=0
    k=0
    for i in range(total_number):
        if i==index_randm[j]:
            line=linecache.getline(file1_in,j+1)
            labels=linecache.getline(label1_in,j+1)
            file0.writelines(line)
            j=j+1
            file1.writelines(labels)
        else:
            # print(i)
            line=linecache.getline(file2_in,k+1)
            labels=linecache.getline(label2_in,k+1)
            file0.writelines(line)
            k=k+1
            file1.writelines(labels)
    print('j:',j)
    print('k:',k)



def step_1(total_num,normal_num):
    # step 1  ------------------------  mix anomalous and normal data -----------------------
    normal_data=Root_folder+'Unique-data/Normal3.csv'
    anomalous_data=Root_folder+'Unique-data/Anomalous3.csv'
    total_num=131909
    normal_num=73663
    mixed_out=Root_folder+'Train&Test/Mixed.csv'
    label_out=Root_folder+'Train&Test/Label-mixed.csv'
    Data_mix(normal_data,anomalous_data,total_num,normal_num,mixed_out,label_out)
    print ('---------- Step_1 finished ')

def step_2(train_number):
    # step 2 --------------- divide mixed data into train and test data -----------------------

    file_in=Root_folder+'Train&Test/Mixed.csv'
    label_in=Root_folder+'Train&Test/Label-mixed.csv'
    file_train=Root_folder+'Train&Test/data_train.csv'
    label_train=Root_folder+'Train&Test/label_train.csv'
    file_test=Root_folder+'Train&Test/data_test.csv'
    label_test=Root_folder+'Train&Test/label_test.csv'
    Extract_Train_Test(file_in,label_in,file_train,label_train,file_test,label_test,number=train_number)
    print ('---------- Step_2 finished ')

def step_3():
    # step 3 ------------------ find key words in URLs -------------------
    file_in=Root_folder+'Train&Test/data_train.csv'
    file_keywords=Root_folder+'Replace-data/keyword.csv'
    smallkey_words,allkeywords=Find_keywords_New(file_in,file_keywords)
    # ------ temp add 
    # print('small key words: ', len(smallkey_words), 'all key words: ',len(allkeywords))

    # ------------------------- Replace words in URLs --------------------------------
    file_in=Root_folder+'Train&Test/data_train.csv'
    file_out=Root_folder+'Replace-data/Replace_train.csv'
    Replace_words_New(file_in,file_out,smallkey_words,allkeywords)
    file_in=Root_folder+'Train&Test/data_test.csv'
    file_out=Root_folder+'Replace-data/Replace_test.csv'
    Replace_words_New(file_in,file_out,smallkey_words,allkeywords)
    print ('---------- Step_3 finished ')

def step_4():
    # step 4 -------------- train the word2vec model (Only use train data)---------------------
    file_in=Root_folder+'Replace-data/Replace_train.csv'
    models_out=Root_folder+'Model-Word2vec/word2vec'
    Word_2vec_Save(file_in,models_out,0,48,100)
    # model=gensim.models.Word2Vec.load(models_out)
    # print (model.most_similar('<'))
    print ('---------- Step_4 finished ')

def step_5():
    # step 5 -------------- encoding --------------------------
        # ----- train 
    file_name=Root_folder+'Replace-data/Replace_train.csv'
    models=Root_folder+'Model-Word2vec/word2vec'
    files_out=Root_folder+'Encode-data/encode_train.csv'
    encode(models,1,file_name,files_out)
    # ----- test 
    file_name=Root_folder+'Replace-data/Replace_test.csv'
    models=Root_folder+'Model-Word2vec/word2vec'
    files_out=Root_folder+'Encode-data/encode_test.csv'
    encode(models,1,file_name,files_out)
    print ('---------- Step_5 finished ')


# ----------------- process the data for fasttext
def fasttext_data(file_name,label_name,output_file):
    label_data=pd.read_csv(label_name,header=None).values
    data_open=open(file_name,'r')
    save_file=open(output_file,'wt')
    index=0
    for line in data_open:
        label=label_data[index][0]
        if label==0:
            new_line='__label__0 '+line
        else:
            new_line='__label__1 '+line
        save_file.writelines(new_line)
        index+=1
    save_file.close()
    data_open.close()
    




if __name__ == "__main__":

    Root_folder='Data/ALL-New-V2/'
    # ----------------------------- step_1: Mix normal and anomalous data
    total_num=131909
    normal_num=73663
    step_1(total_num,normal_num)

    # ----------------------------- step_2: Generate Trian and Test data
    train_number=79146
    step_2(train_number)

    # ----------------------------- step_3: Replace words
    step_3()

    # ----------------------------- step_4: Train Word2vec model
    step_4()

    # ----------------------------- step_5: Encoding data 
    step_5()



    
    

    
    










