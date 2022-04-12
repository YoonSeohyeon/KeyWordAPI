# -*- coding: utf-8 -*-
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.preprocessing import normalize
from konlpy.tag import Komoran
import nltk
import re
# nltk.download()

komoran = Komoran(userdic='user_dict.txt')


def eng_tokenize(sent):
    sent = sent.lower()
    word_tokens = nltk.word_tokenize(sent)
    tokens_pos = nltk.pos_tag(word_tokens)
    NN_words = []
    for word, pos in tokens_pos:
        if 'NN' in pos:
            NN_words.append(word)
    return NN_words



def komoran_tokenize(sent):
    stopword = '아 휴 아이구 아이쿠 아이고 어 나 우리 저희 따라 의해 을 를 에 의 가 으로 로 에게 뿐이다 의거하여 근거하여 입각하여 기준으로 예하면 예를 들면 예를 들자면 저 소인 소생 저희 지말고 하지마 하지마라 다른 물론 또한 그리고 비길수 없다 해서는 안된다 뿐만 아니라 만이 아니다 만은 아니다 막론하고 관계없이 그치지 않다 그러나 그런데 하지만 든간에 논하지 않다 따지지 않다 설사 비록 더라도 아니면 만 못하다 하는 편이 낫다 불문하고 향하여 향해서 향하다 쪽으로 틈타 이용하여 타다 오르다 제외하고 이 외에 이 밖에 하여야 비로소 한다면 몰라도 외에도 이곳 여기 부터 기점으로 따라서 할 생각이다 하려고하다 이리하여 그리하여 그렇게 함으로써 하지만 일때 할때 앞에서 중에서 보는데서 으로써 로써 까지 해야한다 일것이다 반드시 할줄알다 할수있다 할수있어 임에 틀림없다 한다면 등 등등 제 겨우 단지 다만 할뿐 딩동 댕그 대해서 대하여 대하면 훨씬 얼마나 얼마만큼 얼마큼 남짓 여 얼마간 약간 다소 좀 조금 다수 몇 얼마 지만 하물며 또한 그러나 그렇지만 하지만 이외에도 대해 말하자면 뿐이다 다음에 반대로 반대로 말하자면 이와 반대로 바꾸어서 말하면 바꾸어서 한다면 만약 그렇지않으면 까악 툭 딱 삐걱거리다 보드득 비걱거리다 꽈당 응당 해야한다 에 가서 각 각각 여러분 각종 각자 제각기 하도록하다 와 과 그러므로 그래서 고로 한 까닭에 하기 때문에 거니와 이지만 대하여 관하여 관한 과연 실로 아니나다를가 생각한대로 진짜로 한적이있다 하곤하였다 하 하하 허허 아하 거바 와 오 왜 어째서 무엇때문에 어찌 하겠는가 무슨 어디 어느곳 더군다나 하물며 더욱이는 어느때 언제 야 이봐 어이 여보시오 흐흐 흥 휴 헉헉 헐떡헐떡 영차 여차 어기여차 끙끙 아야 앗 아야 콸콸 졸졸 좍좍 뚝뚝 주룩주룩 솨 우르르 그래도 또 그리고 바꾸어말하면 바꾸어말하자면 혹은 혹시 답다 및 그에 따르는 때가 되어 즉 지든지 설령 가령 하더라도 할지라도 일지라도 지든지 몇 거의 하마터면 인젠 이젠 된바에야 된이상 만큼 어찌됏든 그위에 게다가 점에서 보아 비추어 보아 고려하면 하게될것이다 일것이다 비교적 좀 보다더 비하면 시키다 하게하다 할만하다 의해서 연이서 이어서 잇따라 뒤따라 뒤이어 결국 의지하여 기대여 통하여 자마자 더욱더 불구하고 얼마든지 마음대로 주저하지 않고 곧 즉시 바로 당장 하자마자 밖에 안된다 하면된다 그래 그렇지 요컨대 다시 말하자면 바꿔 말하면 즉 구체적으로 말하자면 시작하여 시초에 이상 허 헉 허걱 바와같이 해도좋다 해도된다 게다가 더구나 하물며 와르르 팍 퍽 펄렁 동안 이래 하고있었다 이었다 에서 로부터 까지 예하면 했어요 해요 함께 같이 더불어 마저 마저도 양자 모두 습니다 가까스로 하려고하다 즈음하여 다른 다른 방면으로 해봐요 습니까 했어요 말할것도 없고 무릎쓰고 개의치않고 하는것만 못하다 하는것이 낫다 매 매번 들 모 어느것 어느 로써 갖고말하자면 어디 어느쪽 어느것 어느해 어느 년도 라 해도 언젠가 어떤것 어느것 저기 저쪽 저것 그때 그럼 그러면 요만한걸 그래 그때 저것만큼 그저 이르기까지 할 줄 안다 할 힘이 있다 너 너희 당신 어찌 설마 차라리 할지언정 할지라도 할망정 할지언정 구토하다 게우다 토하다 메쓰겁다 옆사람 퉤 쳇 의거하여 근거하여 의해 따라 힘입어 그 다음 버금 두번째로 기타 첫번째로 나머지는 그중에서 견지에서 형식으로 쓰여 입장에서 위해서 단지 의해되다 하도록시키다 뿐만아니라 반대로 전후 전자 앞의것 잠시 잠깐 하면서 그렇지만 다음에 그러한즉 그런즉 남들 아무거나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 어떻게 만약 만일 위에서 서술한바와같이 인 듯하다 하지 않는다면 만약에 무엇 무슨 어느 어떤 아래윗 조차 한데 그럼에도 불구하고 여전히 심지어 까지도 조차도 하지 않도록 않기 위하여 때 시각 무렵 시간 동안 어때 어떠한 하여금 네 예 우선 누구 누가 알겠는가 아무도 줄은모른다 줄은 몰랏다 하는 김에 겸사겸사 하는바 그런 까닭에 한 이유는 그러니 그러니까 때문에 그 너희 그들 너희들 타인 것 것들 너 위하여 공동으로 동시에 하기 위하여 어찌하여 무엇때문에 붕붕 윙윙 나 우리 엉엉 휘익 윙윙 오호 아하 어쨋든 만 못하다 하기보다는 차라리 하는 편이 낫다 흐흐 놀라다 상대적으로 말하자면 마치 아니라면 쉿 그렇지 않으면 그렇지 않다면 안 그러면 아니었다면 하든지 아니면 이라면 좋아 알았어 하는것도 그만이다 어쩔수 없다 하나 일 일반적으로 일단 한켠으로는 오자마자 이렇게되면 이와같다면 전부 한마디 한항목 근거로 하기에 아울러 하지 않도록 않기 위해서 이르기까지 이 되다 로 인하여 까닭으로 이유만으로 이로 인하여 그래서 이 때문에 그러므로 그런 까닭에 알 수 있다 결론을 낼 수 있다 으로 인하여 있다 어떤것 관계가 있다 관련이 있다 연관되다 어떤것들 에 대해 이리하여 그리하여 여부 하기보다는 하느니 하면 할수록 운운 이러이러하다 하구나 하도다 다시말하면 다음으로 에 있다 에 달려 있다 우리 우리들 오히려 하기는한데 어떻게 어떻해 어찌됏어 어때 어째서 본대로 자 이 이쪽 여기 이것 이번 이렇게말하자면 이런 이러한 이와 같은 요만큼 요만한 것 얼마 안 되는 것 이만큼 이 정도의 이렇게 많은 것 이와 같다 이때 이렇구나 것과 같이 끼익 삐걱 따위 와 같은 사람들 부류의 사람들 왜냐하면 중의하나 오직 오로지 에 한하다 하기만 하면 도착하다 까지 미치다 도달하다 정도에 이르다 할 지경이다 결과에 이르다 관해서는 여러분 하고 있다 한 후 혼자 자기 자기집 자신 우에 종합한것과같이 총적으로 보면 총적으로 말하면 총적으로 대로 하다 으로서 참 그만이다 할 따름이다 쿵 탕탕 쾅쾅 둥둥 봐 봐라 아이야 아니 와아 응 아이 참나 년 월 일 령 영 일 이 삼 사 오 육 륙 칠 팔 구 이천육 이천칠 이천팔 이천구 하나 둘 셋 넷 다섯 여섯 일곱 여덟 아홉 령 영 이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 년 가 한 지 대하 오 말 일 그렇 위하 때문 그것 두 말하 알 그러나 받 못하 일 그런 또 문제 더 사회 많 그리고 좋 크 따르 중 나오 가지 씨 시키 만들 지금 생각하 그러 속 하나 집 살 모르 적 월 데 자신 안 어떤 내 내 경우 명 생각 시간 그녀 다시 이런 이열 앞 보이 번 나 다른 어떻 여자 개 전 들 사실 이렇 점 싶 말 정도 좀 원 잘 통하 놓 오오오 오오 대박 올 헐 엇 아하 오호 엌 잉 엑 으 웩 박박 나나나 이거 처리 저거 그거 헐 최고 신기 완전 정말 정말로 진짜 근데 사실 흠 음 흥 감사 아놔 가 나 다 라 마 바 사 아 자 차 카 타 파 하 아뇨 넵 아니오 넹 녜 녱 듯 | 하이 헬로'
    stopwords= stopword.split(" ")
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    result = hangul.sub('', sent)
    print('한글 정제결과',result)
    word = komoran.nouns(result)
    # for i,v in enumerate(words):
    #     if len(v) < 2:
    #         words.pop(i)
    words = [w for w in word if w not in stopwords] + eng_tokenize(sent)
    return words


sent = ['아이유(IU)의 킬링보이스를 - - - < 라이브로! - 하루 끝, 너의 의미, 스물셋, 밤편지, 팔레트, 가을 아침, 삐삐, Blueming, 에잇, Coin, 라일락 ㅣ 딩고뮤직 코딩너무 즐겁다!! love 2022 아이유 IU 노래모음ㅣBest Songs of IUㅣPlaylist [MV] YOUNHA(윤하) _ 사건의 지평선(Event Horizon)']


def scan_vocabulary(sents, tokenize, min_count=1):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx


def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k: v for k, v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    return dict_to_mat(counter, n_vocabs, n_vocabs)


def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def word_graph(sents, tokenize=None, min_count=2, window=2, min_cooccurrence=2):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    tokens = [tokenize(sent) for sent in sents]
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence)
    return g, idx_to_vocab


def pagerank(x, df=0.85, max_iter=30):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm='l1')
    R = np.ones(A.shape[0]).reshape(-1, 1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R


def textrank_keyword(sents, tokenize, min_count, window, min_cooccurrence, df=0.85, max_iter=30, topk=2):
    # sent = check_word(sents)
    g, idx_to_vocab = word_graph(sents, tokenize, min_count, window, min_cooccurrence)
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx]) for idx in reversed(idxs)]
    return keywords

# sentence = ["안녕하세요 이감국어교육연구소입니다.",
#                    "너무 길어서 힘드네요...어쩔 수 없죠그런데 제목도 깁니다 ㅠ",
#                    "오늘은 문학에서 등장하는 정확히 잡기 힘든 개념어를 이야기해보고자 합니다."
#                    ,"댓글로 질문해주셨던 대화체 독백체로 일단 시작을 해보겠습니다.",
#                    "일단 제목은 더럽게 길지만 요약하자면 일단 다음과 같이 분류할 수 있습니다.",
#                    "1.대화의 형식2.대화체3.독백 = 독백체 = 독백적 발화(내적 독백과 다름) = 독백적 어조2번의 개념이 1번 개념을 포함하고 있으며2번 개념과 3번 개념은 다소 겹칩니다.",
#                    "정리한다면서 더더욱 복잡해지고 있습니다.",
#                    "네 이렇게나 어렵기 때문에 수능에서도 그렇게 엄격하게 출제한 적은 별로 없습니다.",
#                    "일단 상황을 전제해 봅시다.A와 B 두 사람이 있습니다.",
#                    "A가 B에게 말을 하고 B는 그에 대한 반응을 보입니다(대답을 합니다)A는 이 때 화자가 되고 B는 청자가 됩니다.",
#                    "화자가 드러나고 청자가 화자의 말에 대한 반응을 합니다.이건 명백한 ‘대화’의 상황입니다.",
#                    "대화의 상황은 대화의 형식을 빌려~ 라고 선지에서 주로 표현되며이것이 1번의 개념이 됩니다.그러면 1번의 대화와 2번의 대화체는 어떻게 다를까요?겉모습부터 봅시다.",
#                    "대화체라는 녀석한테는 대화와 다르게 ‘체’가 추가되었습니다. ‘체’는 말투처럼 이해하시길 바랍니다.",
#                    "즉, 대화체는 대화를 하는 듯한 말투입니다.그리고 실제 수능에서는 이와 비슷한‘말을 건네는 어투’로 가장 많이 등장합니다.다시 말해서 실제로 대화가 이루어질 필요는 없습니다.",
#                    "다시 말해서 화자가 말을 건네고 있다면 청자의 반응이 존재하지 않더라도,청자의 존재만 뚜렷하게 인정된다면 대화체가 인정됩니다.왜? 실제 청자의 대답이 없더라도 충분히 대화를 하는 듯한 말투가 되기 때문입니다.",
#                    "그러면 대화체면 대화다라는 말은 성립할까요?아닙니다. 아닐수도 있죠. 왜냐하면 청자의 반응이 없더라도 청자의 존재만 인정되면 대화체가 되지만 대화는 반드시 청자의 대답을, 반응을 필요로 하기 때문입니다.",
#                    "그렇다면 대화는 대화체다 라는 말은 어떨까요?성립합니다. 왜냐하면 대화라면 화자도 말을 하고 청자도 그에 대한 반응을 하고 있으니 청자가 당연히 존재하므로 대화체로 인정되기 때문입니다.",
#                    "다시 정리하겠습니다.1.대화체는 대화의 형식을포함합니다. 다만 대화체는 대화의 형식과다릅니다.2. 대화의 형식은 청자의 반응을 반드시 필요로 하지만 대화체는 청자의 반응을 필요로 하지는 않습니다. 화자의 존재만 전제되면 됩니다.3. 말을 건네는 어투도 대화체랑거의 같은 표현입니다..",
#                    "다만 표현 조금 더 단단하게 잡고 간다면 대화체는 말을 건네는 어투일때도 인정이 되고대화적 형식일때도 인정이 되는 겁니다.",
#                    "이제 마지막 개념을 잡아봅시다.독백이나 독백체나 독백적 발화나 독백적 어조나 다 같은 표현으로 생각해도 괜찮아요.독백은 말 그대로 화자가 혼자 말하는 상황 상상하시면 됩니다.그러면 청자는 반드시 필요할까요? 아뇨 없어도 됩니다.따라서 말을 하는 듯하는데 청자가 존재하지 않다면 독백, 독백체로 인정됩니다.",
#                    "아 그러면 청자가 존재하면, 즉 드러나면 독백이 아닌가요?그렇지는 않습니다.다시 말해서 청자가 존재함이 인정되도, 실제로 청자의 반응이 없다면 이는 대화체로도 인정되지만 독백체로도 인정이 됩니다.",
#                    "이제 슬슬 짜증이 나죠. 제가 그래서 2번 3번이 다소 겹친다고 합니다.이 지점때문에 수능 문학 개념에서 그렇게 큰 비중을 차지하지는 못하게 됩니다..그냥 이해하기 쉽게 이렇게 생각합시다.",
#                    "화자가 말을 하고 청자가 존재하되 청자의 반응이 없다면 청자가 존재하기때문에 대화는 아니더라도 대화체가 인정이 된다. 거기에 일단 청자의 반응이 없기 때문에 화자가 혼자 말하고 있고 따라서 독백, 독백체, 독백적 발화, 독백체도 맞는 말이다…하고 잡아주시면 됩니다.",
#                    "아 독백체 진짜 짜증나요! 싶으시면 이렇게 생각하셔도 됩니다.대부분의 시는 독백체입니다.독백체가 아닌 시들은 ‘말’처럼 안 느껴집니다.이런 시들은 어떤 느낌이 드냐면..정말 담담한 느낌이 드는 경우가 많습니다.그래서 주로 평서형 종결 어미와 명사형 종결 어미들이 쓰입니다.이런 경우가 아니라면 보통 정말 ‘말’을 하는 듯한 인상을 주고, 그러면 독백체라고 인정합니다.",
#                    "다시 정리합니다.1.대화체와 독백체가 동시에 쓰일 수 있습니다.2.실제로 많은 시들이 독백체입니다.도움이 되셨길 바랍니다"]
# #
print(textrank_keyword(sent, komoran_tokenize, 1, 2,2))