import nltk
from nltk.stem.porter import PorterStemmer

def extract_candidates(text):

    GRAMMAR_EN = """  NP:
{<NN.*|JJ>*<NN.*>}"""   # Adjective(s)(optional) + Noun(s)
    keyphrase_candidate = set()
    

    np_parser = nltk.RegexpParser(GRAMMAR_EN)  # Noun phrase parser
    
    tag = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    
    trees = np_parser.parse_sents(tag)  # Generator with one tree per sentence
    #print(text)

    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))
    
    #print(keyphrase_candidate)
    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 4}
    #print(keyphrase_candidate)
  
    return keyphrase_candidate


def get_fscore(pred,label):
    
    new_pred=[]
    
    for e in pred:
        e = e.replace('-',' ')
        c=''
        e = e.split(' ')
        if len(e)>4:
            continue
        for q in e:
            c = c + ' ' + porter_stemmer.stem(q)
        if c.strip() not in new_pred:
            new_pred.append(c.strip())

    tmp=[]
    
    for e in new_pred:
        flg=0 
        for w in tmp:
            if w in e:
                flg=1
                break
        if flg==0:
            tmp.append(e)
    new_pred = tmp
    
    
    new_pred = new_pred[:min(10,len(new_pred))]
    pred=new_pred
    
    precision=0
    recall=0
    for e in label:
        if e in pred:
            recall += 1
            precision += 1
        
    if precision==0:
        return 0
    precision /= len(pred)
    recall /= len(label)

    return 2*precision*recall/(precision+recall)
