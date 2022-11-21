
import dill

voc_path = '../data1/voc_final.pkl'
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
diag = diag_voc.idx2word
pro = pro_voc.idx2word
med = med_voc.idx2word

def medication_classify():
    real_med = [2, 4, 5, 6, 7, 8, 11, 12, 13, 22, 1, 3, 26, 28, 52, 44, 39, 42, 32, 41, 27, 63, 20, 37, 101, 14]
    real_med_code = []
    pred_med = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 20, 22, 28, 32, 37, 44, 52, 82]
    pred_med_code = []
    for med_code in med.keys():
        for real in real_med:
            if med_code == real:
                real_med_code.append(med[real])
        for pred in pred_med:
            if med_code == pred:
                pred_med_code.append(med[pred])
    correct = list(set(real_med_code).intersection(set(pred_med_code)))
    print('真实药物组合的编码{0}，\n长度为{1}'.format(real_med_code, len(real_med_code)))
    print('预测药物组合的编码{0}，\n长度为{1}'.format(pred_med_code, len(pred_med_code)))
    print("统计正确的药物{0}，\n长度为{1}".format(correct, len(correct)))

def diagnisis_classify():
    rare_diagnosis = dill.load(open('rare_diagnosis.pkl', 'rb'))
    diagnisis = [352, 228, 551, 282, 321, 938, 13, 942, 64, 1781, 307, 35, 209, 18, 9, 414, 20, 269, 51, 313, 254, 792, 384, 785, 25, 1668, 399, 7, 69, 172, 49]
    diagnosis_code = []
    rare = []
    for diag_ in diag.keys():
        for real in diagnisis:
            if diag_ == real:
                diagnosis_code.append(diag[real])
                for rare_diag in rare_diagnosis.keys():
                    if rare_diag == real:
                        rare.append(diag[real])

    print('诊断的编码{}'.format(diagnosis_code))
    print('稀有诊断结果{}'.format(rare))

def hamming_data():
    diagnosis = []
    diagnosis_code = [146, 14, 51, 25, 147]
    procedure = []
    procedure_code = [67, 7, 52, 12, 68]
    medication = []
    medication_code = [0, 22, 6, 45, 70, 12]
    prior_medication_code = [0, 1, 2, 3, 4, 6, 12, 22, 26]
    prior_medication = []
    for med_code in med.keys():
        for real in medication_code:
            if med_code == real:
                medication.append(med[real])
        for real0 in prior_medication_code:
            if med_code == real0:
                prior_medication.append(med[real0])
    for diag_code in diag.keys():
        for diag_ in diagnosis_code:
            if diag_code == diag_:
                diagnosis.append(diag[diag_])
    for pro_code in pro.keys():
        for pro_ in procedure_code:
            if pro_code == pro_:
                procedure.append(pro[pro_])
    print('诊断：', diagnosis)
    print('治疗：', procedure)
    print('用药：', medication)
    print('先验用药：', prior_medication)

def main():
    medication_classify()
    # diagnisis_classify()

    # hamming_data()




if __name__ == '__main__':
    main()