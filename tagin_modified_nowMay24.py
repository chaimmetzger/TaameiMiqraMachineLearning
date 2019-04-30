from tf.fabric import Fabric
import collections
import sys
# https://etcbc.github.io/bhsa/features/hebrew/4b/features/comments/g_lex_utf8.html

TF = Fabric(locations='/home/chaim/github/text-fabric-data', modules=['hebrew/etcbc4c'])
#TF = Fabric(locations='c:/josh/text-fabric-data/text-fabric-data', modules=['hebrew/etcbc4c'])
api = TF.load('sp lex g_word g_word_utf8 trailer_utf8 ls typ rela function qere_utf8 qere')
api.makeAvailableIn(globals())

F = api.F
T = api.T
C = api.C
L = api.L

#print(sorted(T.formats))




def print_original_words():

    for i in range(1, 12):
        print(api.T.text([i], 'text-orig-full'))
# for w in F.otype.s('word'):
#     word, part_of_speech = F.g_word.v(w), F.sp.v(w)
#     print(word, part_of_speech)
#     if w == 14:
#         break

import sys

# uncomment this if want to print to screen
# outfile=sys.stdout

# uncomment this if want to print to file
#outfile = open('POSAndTaamPairsForAllOfTanakh_1.txt', 'w')

GRANULARITY = "high" # low / high
LEXICAL = True
POS_FOR_PARTIAL_WORD = 0
POS_FOR_FINAL_WORD = 1
POS_AGGREGATE_FOR_WORD = 2

if len(sys.argv) > 1:
    if sys.argv[1] == '-aggregate':
        pos_behavior = POS_AGGREGATE_FOR_WORD
    elif sys.argv[1] == '-partial':
        pos_behavior = POS_FOR_PARTIAL_WORD
    else:
        pos_behavior = POS_FOR_PARTIAL_WORD
else:
    pos_behavior = POS_AGGREGATE_FOR_WORD


# alternatively, assign it in code, as below
pos_behavior = POS_FOR_PARTIAL_WORD
#pos_behavior = POS_AGGREGATE_FOR_WORD

outfile_name = 'POSAndTaamPairsForAllOfTanakh18'

if pos_behavior == POS_AGGREGATE_FOR_WORD:
    outfile_name += '_word_aggregate'
elif pos_behavior == POS_FOR_PARTIAL_WORD:
    outfile_name += '_wordbreak'

from collections import defaultdict
dictTrupToPos = defaultdict(set)
dictPosToTrup = defaultdict(set)

outfile_name += '.txt'
outfile = open(outfile_name, 'w', encoding='utf8')
print('Output file name:', outfile_name)

def extract_trup(book: str, s: str, trailer: str) -> str:
    if trailer == '־':
        trup = 'MAQEF'
    elif '75' in s or '25' in s:  # 25 occurs in Vayikra 21:18
        trup = 'SILLUQ'
    elif '92' in s:
        trup = 'ETNACHTA'
    elif '73' in s:
        trup = 'TIPCHA'
    elif '80' in s:
        trup = 'ZAKEFKATON'
    elif '01' in s:
        trup = 'SEGOLTA'
    elif '74' in s:
        if trailer == '׀ ':
            trup = 'MUNACHLEGARMEIH'
        else:
            trup = 'MUNACH'
    elif '71' in s:
        trup = 'MERCHA'
        start_pasuk = True
    elif '81' in s:
        trup = 'REVIA'
    elif '70' in s:
        trup = 'MAHPACH'

        if book in ('Job', 'Proverbs', 'Psalms') and trailer == '׀ ':
                trup = 'MAHPACHLEGARMEIH'
    elif '03' in s:
        trup = 'PASHTA'
    elif '85' in s:
        trup = 'ZAKEFGADOL'
    elif '94' in s:
        trup = 'DARGA'
    elif '91' in s:
        trup = 'TEVIR'
    elif '63' in s:
        trup = 'KADMA'
    elif '61' in s:
        trup = 'GERESH'
    elif '62' in s:
        trup = 'GERSHAYIM'
    elif '02' in s:
        trup = 'ZARQA'
    elif '10' in s:
        trup = 'YETIV'
    elif '14' in s or '44' in s:
        trup = 'TELISHAGEDOLA'
    elif '04' in s or '24' in s:
        trup = 'TELISHAKETANA'
    elif '83' in s:
        trup = 'PAZER'
    elif '11' in s:
        trup = 'GERESHMUKDAM'
    elif '65' in s:
        trup = 'SHALSHELET'
        if book in ('Job', 'Proverbs', 'Psalms') and trailer == '׀ ':
                trup = 'SHALSHELETLEGARMEIH'
    elif '72' in s:
        trup = 'MERCHAKEFULA'
    elif '93' in s:
        trup = 'GALGAL'
    elif '84' in s:
        trup = 'KARNEIPARA'
    elif '13' in s:
        trup = 'DECHI'
    elif '60' in s:
        trup = 'OLEH'
    elif '64' in s:
        trup = 'ILUY'
    elif '82' in s:
        trup = 'TZINORIT'
    elif '33' in s:
        trup = 'PASHTA'  # sometimes claim is KADMA?
    elif '45' in s:  # meteg;
        s = s.replace('45', '')  # strip it out, will confuse
    elif '95' in s:  # another meteg?
        s = s.replace('95', '')  # strip it out, will confuse
    elif '35' in s:  # another meteg?
        s = s.replace('35', '')  # strip it out, will confuse
    elif '52' in s:  # dots over letters to indicate possible absence
        s = s.replace('52', '')  # strip it out, will confuse
    elif not any(c.isdigit() for c in s):
        s = ''  # there was a maqef. for now, just eat the digit

    else:
        print('-------------------------------------Still needs processing: ', word)
        sys.exit(1)
    trup = ''

    return s, trup

def print_trup_pos():
    for verse in F.otype.s('verse'):
        book = T.sectionFromNode(verse)[0]
        print(T.sectionFromNode(verse)[0], T.sectionFromNode(verse)[1], str(T.sectionFromNode(verse)[2]) + ':', end=' ', file=outfile)
        pasuk = []
        s = ''  # type: str
        aggregate_trup = []
        aggregate_pos = []
        aggregate_pos_subtype = []
        aggregate_lex = []

        parts = L.d(verse, 'half_verse')
        for hv, half_verse in enumerate(parts):
            words = L.d(half_verse, 'word')

            for w in words:
                word, part_of_speech, lex, trailer = F.g_word.v(w), F.sp.v(w), F.g_word_utf8.v(w), F.trailer_utf8.v(w)
                pos_subtype, simple_lex = F.ls.v(w), F.lex.v(w)
                qere_utf8, qere = F.qere_utf8.v(w), F.qere.v(w)
                if qere is not None:
                    lex = qere_utf8
                    word = qere

                trupsymbols = '֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯׀'
                for c in trupsymbols:
                    lex = lex.replace(c, '')

                s += word

                # acquire features of phrases
                phrase = L.u(w, 'phrase')[0]
                clause = L.u(w, 'clause')[0]

                p_typ, p_function = F.typ.v(phrase), F.function.v(phrase)
                c_typ, c_function = F.typ.v(clause), F.function.v(clause)

                s, trup = extract_trup(book, s, trailer)

                if pos_behavior == POS_FOR_PARTIAL_WORD:
                    # even though this is a subword unit, we want to output everything
                    # 8 items in the tuple
                    pasuk.append((trup, part_of_speech, pos_subtype, lex, hv, p_typ, p_function, c_typ))

                    # perhaps this is also the end of an actual full word,
                    # in which case we should also output a tuple full of WORDBREAK
                    if s != '' and s[-1] != '-':
                        pasuk.append(tuple(['WB'] * 8))

                    s = ''
                elif pos_behavior == POS_AGGREGATE_FOR_WORD:
                    # we don't output anything yet, since this is a sub-word unit
                    # just aggregate it until we reach the actual end of the word
                    if trup != '':
                        aggregate_trup.append(trup)
                    aggregate_pos.append(part_of_speech)
                    if pos_subtype != 'none':
                        aggregate_pos_subtype.append(pos_subtype)
                    aggregate_lex.append(lex)

                    # perhaps this is also the end of an actual full word,
                    # in which case we should also output all we have aggregated
                    if s != '' and s[-1] != '-':
                        pasuk.append(('_'.join(aggregate_trup), '_'.join(aggregate_pos),
                                      '_'.join(aggregate_pos_subtype), '_'.join(aggregate_lex),
                                      hv, p_typ, p_function, c_typ))
                        aggregate_trup = []
                        aggregate_pos = []
                        aggregate_lex = []
                        s = ''
                # end for w in words
        # end hv, half_verse in enumerate(parts)
        print(pasuk, file=outfile)
        allTrup = tuple(t[0] for t in pasuk)
        allPos = tuple(t[1] for t in pasuk)
        dictTrupToPos[allTrup].add(allPos)
        dictPosToTrup[allPos].add(allTrup)
        # end for verse in F.otype.s('verse'):




    # f2 = open('multivalence_trup.txt', 'w')
    # print('dictTrupToPos length:', len(dictTrupToPos), file=f2)
    # print('dictPosToTrup length:', len(dictPosToTrup), file=f2)
    # print("Therefore, there are more POS sequences than TRUP sequences.", file=f2)
    # print("From POS --> TRUP is easier than from TRUP --> POS.", file=f2)
    # print("Looking at TRUP --> POS", file=f2)
    # for key in dictTrupToPos.keys():
    #     value = dictTrupToPos[key]
    #     if len(value) > 1:
    #         print('--------------------', file = f2)
    #         print(len(value), key, file=f2)
    #         for pos_sequence in value:
    #             print('\t', pos_sequence, file=f2)
    #
    # f2.close()
    #
    # f2 = open('multivalence_pos.txt', 'w')
    # print('dictTrupToPos length:', len(dictTrupToPos), file=f2)
    # print('dictPosToTrup length:', len(dictPosToTrup), file=f2)
    # print("Therefore, there are more POS sequences than TRUP sequences.", file=f2)
    # print("Looking at POS --> TRUP:", file=f2)
    # for key in dictPosToTrup.keys():
    #     value = dictPosToTrup[key]
    #     if len(value) > 1:
    #         print('--------------------', file=f2)
    #         print(len(value), key, file=f2)
    #         for pos_sequence in value:
    #             print('\t', pos_sequence, file=f2)
    #
    # f2.close()


print_trup_pos()
