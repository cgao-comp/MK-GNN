
import dill
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import hamming_loss

def pkl2excel(filename):

    path = filename
    files = os.listdir(path)
    for file in files:
        # data = dill.load(open(filename + '\\' + file, 'rb'))
        # for i in range(len(data)):
        #     data[i] = data[i].detach().numpy()
        f = open(filename + '\\' + file,'rb')
        data = pickle.load(f)
        pd.DataFrame(data).to_excel(filename + '\\' + file + '.xlsx')


    # data = dill.load(open(filename, 'rb'))
    # data25 = data[25].detach().numpy()
    #
    # print(data25)
    # for i in range(len(data25)):
    #     head = data25[i]
    #     mat = np.zeros((len(head), len(head)))
    #     for j in range(len(head)):
    #         for k in range(len(head[j])):
    #             mat[j][k] = head[j][k]
    #
    #     pd.DataFrame(mat).to_excel(filename + str(i) + '.xlsx')

    print("转换完成！！！")


def deal_data(flag, saveFileName, visit, per):
    data_file1 = 'E:\whq\mutual information\data\PKL\KDD\\records_final.pkl'
    # data_file1 = 'E:\论文\mutual information\data\PKL\\all_data(visit_lt1)\\records_final.pkl'
    data = dill.load(open(data_file1, 'rb'))
    # data = data[:2]
    # print(data)

    patient_num = len(data)
    diagnoses = {}  # 每种疾病的所有可能用药
    diagnisisVisistNum = {}  # 每一种疾病被诊断的次数
    for patient, records in enumerate(data):
        visit_num = len(records)
        # print("当前患者就诊了{}次".format(visit_num))
        for adm, each_record in enumerate(records):
            # diagnosis_num = len(each_record[0])
            for diag in each_record[flag]:  # each_record[0] ----> 提取诊断信息； each_record[1] ----> 提取治疗信息；
                if diag in diagnoses.keys():
                    diagnoses[diag] = diagnoses[diag] + each_record[2]
                else:
                    diagnoses[diag] = each_record[2]

                if diag in diagnisisVisistNum.keys():
                    diagnisisVisistNum[diag] = diagnisisVisistNum[diag] + 1
                else:
                    diagnisisVisistNum[diag] = 1

    # print(diagnoses[2])
    # print(diagnisisVisistNum[0])
    med = []
    for key in diagnoses.keys():  # 计算所有的药物种类
        med = med + diagnoses[key]
    # print(len(set(med)))

    diag_num = len(diagnoses.keys())
    med_num = len(set(med))
    if flag == 0:
        print('诊断统计')
    elif flag == 1:
        print('治疗统计')
    print(diag_num)
    print(med_num)
    diag_med = np.zeros((diag_num, med_num))  # 构建疾病与用药的矩阵，相应位置的数字表示再被诊断为疾病i时，j种药物使用的次数

    for key in diagnoses.keys():
        for value in diagnoses[key]:
            diag_med[key][value] += 1
    # print(diag_med)

    # pd.DataFrame(diag_med).to_excel("diag_med.xlsx")
    # dill.dump(diag_med, open("diag_med.pkl", "wb"))
    #
    # pd.DataFrame(diagnisisVisistNum, index=[0]).to_excel("diagnisisVisistNum.xlsx")
    # dill.dump(diagnisisVisistNum, open("diagnisisVisistNum.pkl", "wb"))

    new_diagnisisVisistNum = {}
    rare_diagnosis = {}  # 稀有疾病（疑难杂症）
    for key1 in diagnisisVisistNum.keys():
        if diagnisisVisistNum[key1] >= visit:
            new_diagnisisVisistNum[key1] = diagnisisVisistNum[key1] * per  # 80% 的确诊
        elif diagnisisVisistNum[key1] < visit:
            rare_diagnosis[key1] = diagnisisVisistNum[key1]

    # pd.DataFrame(new_diagnisisVisistNum, index=[0]).to_excel("diagnisisVisistNum_del.xlsx")
    dill.dump(rare_diagnosis, open('rare_diagnosis.pkl', 'wb'))

    saveDiagMed = {}  # 对于确诊次数10次以上的疾病，如果用药的次数超过80%的确诊次数，作为该疾病可能的用药选择保存下来
    for i in range(diag_num):
        med = []
        if i in new_diagnisisVisistNum.keys():
            for j in range(med_num):
                if diag_med[i][j] > new_diagnisisVisistNum[i]:
                    med.append(j)
            if len(med) > 1:
                saveDiagMed[i] = med

    # 疑难杂症的用药统计
    rare_med = np.zeros((diag_num, med_num))
    for diag in range(diag_num):
        for rare_diag in rare_diagnosis.keys():
            if rare_diag == diag:
                for med in range(med_num):
                    rare_med[rare_diag][med] = diag_med[diag][med]
    pd.DataFrame(rare_med).to_excel("rare_med.xlsx")

    # print(saveDiagMed[0])
    dill.dump(saveDiagMed, open(saveFileName + '_' + str(visit) + '_' + str(per) + ".pkl", "wb"))
    # pd.DataFrame(saveDiagMed, index=[0]).to_excel("saveDiagMed.xlsx")

def FuseOriginData(diagFile, proFile):
    MedData1 = dill.load(open(diagFile, "rb"))
    MedData2 = dill.load(open(proFile, "rb"))
    # print(MedData.keys())
    OriginData = dill.load(open('E:\whq\mutual information\data\PKL\KDD\\records_final.pkl', "rb"))
    for patient, records in enumerate(OriginData):
        for adm, each_record in enumerate(records):
            impossibleMed1 = []
            impossibleMed2 = []
            for diag in each_record[0]:
                if diag in MedData1.keys():
                    impossibleMed1 = impossibleMed1 + MedData1[diag]
            each_record.append(list(set(impossibleMed1)))
            for pro in each_record[1]:
                if pro in MedData2.keys():
                    impossibleMed2 = impossibleMed2 + MedData2[pro]
            each_record.append(list(set(impossibleMed2)))

    dill.dump(OriginData, open('diag_' + proFile, "wb"))

def look(file):
    data = dill.load(open(file, 'rb'))
    print('查看先验信息数据')

# 相似度矩阵
def build_similarity():

    adj = dill.load(open("E:\whq\mutual information\data\ehr_adj_final.pkl", "rb"))
    num_node = adj.shape[0]
    similarity = np.zeros(shape=(num_node, num_node))
    print('similarity', type(similarity))
    print(similarity.shape)
    for i in range(num_node):
        degree_i = sum(adj[i])
        for j in range(num_node):
            degree_j = sum(adj[j])
            degree_comm = float(0)
            for num in range(num_node):
                # if (adj[i][num] == 1) and (adj[j][num] == 1):
                if (adj[i][num] == 1) and (adj[j][num] == 1) and (adj[i][j] != 0):
                    degree_comm += 1.0
            similarity[i][j] = round((2 * degree_comm) / (degree_i + degree_j),2)
    # similarity = similarity - np.eye(num_node)

    # for i in range(num_node):  # 相似度阈值的筛选
    #     for j in range(num_node):
    #         if similarity[i][j] < 0.9:
    #             similarity[i][j] = 0

    dill.dump(similarity, open('similarity_adj.pkl', 'wb'))

def lookRarePatient():

    diag_data = dill.load(open('rare_diagnosis.pkl', 'rb'))
    data = dill.load(open('E:\whq\mutual information\data\PKL\KDD\\records_final.pkl', "rb"))
    print('患者人数为{}'.format(len(data)))
    OriginData1 = dill.load(open('E:\whq\mutual information\data\PKL\不同阈值下的先验信息准确度5_hamming loss\\records_hamming_5_0.73.pkl', "rb"))
    OriginData2 = dill.load(open('E:\whq\mutual information\data\PKL\不同阈值下的先验信息准确度5_hamming loss\\records_hamming_5_0.73.pkl', "rb"))
    rare_diagnosis = set(diag_data.keys())
    i_ = 0
    for i in range(len(OriginData1)):
        j_ = 0
        for j in range(len(OriginData1[i-i_])):
            # print(patient[0])
            if len(rare_diagnosis.intersection(set(OriginData1[i-i_][j-j_][0]))) == 0:
                del OriginData1[i-i_][j-j_]
                j_ = j_ + 1
        if len(OriginData1[i-i_]) == 0:
            del OriginData1[i-i_]
            i_ = i_ + 1
    print('患有稀有疾病的患者人数为{}'.format(len(OriginData1)))
    dill.dump(OriginData1, open("hamming_loss_rare_diagnosis_records_1_0.73.pkl", "wb"))

    i_ = 0
    for i in range(len(OriginData2)):
        j_ = 0
        for j in range(len(OriginData2[i - i_])):
            # print(patient[0])
            if len(rare_diagnosis.intersection(set(OriginData2[i - i_][j - j_][0]))) != 0:
                del OriginData2[i - i_][j - j_]
                j_ = j_ + 1
        if len(OriginData2[i - i_]) == 0:
            del OriginData2[i - i_]
            i_ = i_ + 1
    print('不患有稀有疾病的患者人数为{}'.format(len(OriginData2)))
    dill.dump(OriginData2, open("hamming_loss_diagnosis_records_1_0.73.pkl", "wb"))


def deal_data_static(visit, per):
    data_file1 = 'E:\whq\mutual information\data\PKL\KDD\\records_final.pkl'
    data = dill.load(open(data_file1, 'rb'))
    # split_point = int(len(data) * 2 / 3)
    # data = data[:split_point]
    diagnoses = {}  # 每种疾病的所有可能用药
    diagnisisVisistNum = {}  # 每一种疾病被诊断的次数
    for patient, records in enumerate(data):
        for adm, each_record in enumerate(records):
            for diag in each_record[0]:  # each_record[0] ----> 提取诊断信息； each_record[1] ----> 提取治疗信息；
                if diag in diagnoses.keys():
                    diagnoses[diag] = diagnoses[diag] + each_record[2]
                else:
                    diagnoses[diag] = each_record[2]

                if diag in diagnisisVisistNum.keys():
                    diagnisisVisistNum[diag] = diagnisisVisistNum[diag] + 1
                else:
                    diagnisisVisistNum[diag] = 1
    med = []
    for key in diagnoses.keys():  # 计算所有的药物种类
        med = med + diagnoses[key]

    diag_num = len(diagnoses.keys())
    med_num = len(set(med))
    diag_med = np.zeros((diag_num, med_num))  # 构建疾病与用药的矩阵，相应位置的数字表示再被诊断为疾病i时，j种药物使用的次数

    for key in diagnoses.keys():
        for value in diagnoses[key]:
            diag_med[key][value] += 1

    new_diagnisisVisistNum = {}
    for key1 in diagnisisVisistNum.keys():
        if diagnisisVisistNum[key1] >= visit:
            new_diagnisisVisistNum[key1] = diagnisisVisistNum[key1] * per  # 80% 的确诊

    saveDiagMed = {}  # 对于确诊次数10次以上的疾病，如果用药的次数超过80%的确诊次数，作为该疾病可能的用药选择保存下来
    for i in range(diag_num):
        med = []
        if i in new_diagnisisVisistNum.keys():
            for j in range(med_num):
                if (diag_med[i][j] >= new_diagnisisVisistNum[i]) and (diag_med[i][j] <= diagnisisVisistNum[i]):
                    med.append(j)
            if len(med) > 1:
                saveDiagMed[i] = med
    return saveDiagMed

def FuseOriginData_static(MedData1):
    OriginData = dill.load(open('E:\whq\mutual information\data\PKL\KDD\\records_final.pkl', "rb"))
    for patient, records in enumerate(OriginData):
        for adm, each_record in enumerate(records):
            impossibleMed1 = []
            impossibleMed2 = []
            for diag in each_record[0]:
                if diag in MedData1.keys():
                    impossibleMed1 = impossibleMed1 + MedData1[diag]
            # each_record.append(list(set(impossibleMed1))) # 去除重复
            for i in range(len(impossibleMed1)):
                if impossibleMed1[i] not in impossibleMed2:
                    impossibleMed2.append(impossibleMed1[i])
                # if i != (len(impossibleMed1)-1):
                #     for j in range(i+1, len(impossibleMed1)):
                #         if impossibleMed1[i] == impossibleMed1[j]:
                #             impossibleMed1[j] = 1000000000
            # for item in impossibleMed1:
            #     if item == 1000000000:
            #         impossibleMed1.remove(item)
            each_record.append(impossibleMed2)
    return OriginData

def bestPriorInformation(data):
    all_static_accuracy = 0
    for idx1, patient in enumerate(data):
        current_static_accuracy = 0
        for idx2, adm in enumerate(patient):
            purity = list(set(adm[2]).intersection(set(adm[3])))
            impurity1 = list(set(adm[3]).difference(set(adm[2])))
            impurity2 = list(set(adm[2]).difference(set(adm[3])))
            current_static_accuracy += (len(purity) / (len(purity) + len(impurity1) + len(impurity2)))
        # print("第{0}位患者先验信息统计准确性为：{1}".format(idx1, current_static_accuracy/(idx2+1)))
        all_static_accuracy += (current_static_accuracy / (idx2+1))
    # print("阈值为{0},患者先验信息统计准确性为：{1}".format(per, all_static_accuracy/(idx1+1)))
    return all_static_accuracy / (idx1+1)

def hamming(data):
    hamming_score = 0
    for idx1, patient in enumerate(data):
        hamming_score_ = 0
        for idx2, adm in enumerate(patient):
            purity = list(set(adm[2]).intersection(set(adm[3])))
            impurity1 = list(set(adm[3]).difference(set(adm[2])))
            impurity2 = list(set(adm[2]).difference(set(adm[3])))
            hamming_score_ += (len(impurity1) + len(impurity2)) / len(adm[2])
            # print('Hamming loss is {}'.format(hamming_score))
        hamming_score += (hamming_score_ / (idx2+1))

    hamming_final_score = (hamming_score / len(data))
    return hamming_final_score

def main():
    op = True
    while (op != 0):
        print("请输入要进行的操作，0：结束操作；1：pkl文件转Excell文件；2：debug查看先验信息变量；3：统计先验信息；4：计算相似度矩阵; 5: 生成疑难杂症数据；6：不同阈值下的先验信息准确性")
        op = int(input())
        if op == 1:
            print('请输入要转化的文件路径：')
            fileName = str(input())
            pkl2excel(fileName)  # pkl文件转Excell文件
        elif op == 2:
            look("diag_Pro_Med_10_0.5.pkl")  # debug 查看变量
        elif op == 3:
            flag = 10
            while (flag != 3):
                print('请输入要统计的类型（0表示诊断，1表示治疗,2表示生产最终加入先验信息的文件，3表示结束）：')
                flag = int(input())
                if flag == 0:
                    print('请输入确诊次数阈值：')
                    diag_visit = int(input())
                    print('请输入确诊筛选百分比：')
                    diag_per = float(input())
                    print("诊断数据统计中...")
                    deal_data(flag, 'Diag_Med', diag_visit, diag_per)

                elif flag == 1:
                    print('请输入治疗次数阈值：')
                    pro_visit = int(input())
                    print('请输入治疗筛选百分比：')
                    pro_per = float(input())
                    print("治疗程序数据统计中...")
                    deal_data(flag, 'Pro_Med', pro_visit, pro_per)

                elif flag == 2:
                    diagFile = 'Diag_Med' + '_' + str(diag_visit) + '_' + str(diag_per) + ".pkl"
                    proFile = 'Pro_Med' + '_' + str(pro_visit) + '_' + str(pro_per) + ".pkl"
                    print("最终结果生成中...")
                    FuseOriginData(diagFile, proFile)

                elif flag == 3:
                    print('先验信息统计完成！')
            print("结束先验信息统计！")
        elif op == 4:
            build_similarity()
        elif op == 5:
            lookRarePatient()
        elif op == 6:
            visit = 1
            per_list = np.arange(0, 1.01, 0.01)
            dic = {}
            for per in per_list:
                print("先验信息计算中....")
                diag_data = deal_data_static(visit, per)
                fuse_data = FuseOriginData_static(diag_data)
                # pd.DataFrame(fuse_data).to_excel('hammingloss1_0.xlsx',index=0)
                dill.dump(fuse_data, open('records_hamming_' + str(visit) +'.pkl', 'wb'))
                # static = bestPriorInformation(fuse_data)
                static = hamming(fuse_data)
                dic[per] = static
                print("阈值为{0},患者先验信息统计准确性为：{1}".format(per, static))
            pd.DataFrame(dic, index=[0]).to_excel('不同阈值下的先验信息hamming.xlsx')


if __name__ == '__main__':
    main()