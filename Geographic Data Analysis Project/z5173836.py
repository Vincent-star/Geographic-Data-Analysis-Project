import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # df1 = pd.read_csv("exposure.csv", sep=';', encoding="ISO-8859-1")
    df1 = pd.read_csv(exposure, sep=';', encoding="ISO-8859-1")

    # df1_2 = pd.read_csv("Countries.csv", encoding="ISO-8859-1")
    df1_2 = pd.read_csv(countries, encoding="ISO-8859-1")

    df1.columns = map(str.capitalize, df1.columns)
    # Drop all rows from the "exposure" dataset without country name
    change_name = {'Korea DPR': 'North Korea',
                   'Korea, North': 'North Korea',
                   'Korea Republic of': 'South Korea',
                   'Korea, South': 'South Korea',
                   'United States of America': 'US',
                   'United States': 'US',
                   'Viet Nam': 'Vietnam',
                   'Cabo Verde': 'Cape Verde',
                   'Brunei Darussalam': 'Brunei',
                   'Lao PDR': 'Laos',
                   'North Macedonia': 'Macedonia',
                   'Moldova Republic of': 'Moldova',
                   'Russian Federation': 'Russia',
                   'Eswatini': 'Swaziland',
                   "Côte d'Ivoire": 'Ivory Coast',
                   'Republic of the Congo': 'Congo',
                   'Democratic Republic of the Congo': 'Congo DR',
                   'Palestinian Territory': 'Palestine'}
    df1['Country'] = df1['Country'].replace(change_name.keys(), change_name.values())
    df1_2['Country'] = df1_2['Country'].replace(change_name.keys(), change_name.values())
    # Join the two datasets (exposure.csv and Countries.csv) based on the "country" columns in the datasets, keeping the
    # rows as long as there is a match between the country columns of both dataset
    df1 = pd.merge(left=df1, right=df1_2, how='inner')
    # keep only a single country column
    df1 = df1[df1.Country.notna()]
    # set the index of the resultant dataframe as 'Country'
    df1 = df1.set_index('Country')
    # sort the dataset by the index (ascending)
    df1 = df1.sort_index(ascending=True)
    #################################################

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    def calculate_avg(counrty_name, latitude, longitude):
        cities_info_list = df2_dict[counrty_name].split("|||")
        if latitude:
            latitude_list = []
            for i in cities_info_list:
                temp1 = i.split(',')[2]
                latitude_list.append(float(temp1.split(':')[1]))
            result = sum(latitude_list) / len(latitude_list)
            return result
        if longitude:
            longitude_list = []
            for i in cities_info_list:
                temp1 = i.split(',')[3]
                longitude_list.append(float(temp1.split(':')[1]))
            result = sum(longitude_list) / len(longitude_list)
            return result

    df2 = df1.copy()
    df2_dict = df2['Cities'].to_dict()
    df2_countries = df2_dict.keys()
    avg_latitude_list = []
    avg_longitude_list = []
    for i in df2_countries:
        avg_latitude = calculate_avg(i, True, False)
        avg_latitude_list.append(avg_latitude)
        avg_longitude = calculate_avg(i, False, True)
        avg_longitude_list.append(avg_longitude)
    df2['avg_latitude'] = avg_latitude_list
    df2['avg_longitude'] = avg_longitude_list

    # print(df_dict['North Korea'])
    #################################################

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    def calculate_distance(lat_Wuhan, lon_Wuhan, lat_country, lon_country):

        lon1, lat1, lon2, lat2 = map(np.radians, [float(lon_Wuhan), float(lat_Wuhan), float(lon_country), float(lat_country)])

        lon_distance = lon2 - lon1
        lat_distance = lat2 - lat1

        a = np.sin(lat_distance/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon_distance/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        earth_R = 6373
        distance = c * earth_R
        # distance = round(distance, 5)
        return distance

    df3 = df2.copy()
    distance_list = []
    df3_temp = df3['Cities'].to_dict()
    df3_countries = df3_temp.keys()
    # for i in range(len(df3_countries)):
    for i in df3_countries:

        distance = calculate_distance(30.5928, 114.3055, df3.loc[i, 'avg_latitude'], df3.loc[i, 'avg_longitude'])

        distance_list.append(distance)

    df3['distance_to_Wuhan'] = distance_list
    df3 = df3.sort_values(by='distance_to_Wuhan', ascending=True)
    # df3[['distance_to_Wuhan']] = df3[['distance_to_Wuhan']].astype(str)


    #################################################

    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df4 = df2.copy()
    df4 = df4.reset_index()
    cont = pd.read_csv(continents, encoding="ISO-8859-1")

    change_name = {'Korea DPR': 'North Korea',
                   'Korea, North': 'North Korea',
                   'Korea Republic of': 'South Korea',
                   'Korea, South': 'South Korea',
                   'United States of America': 'US',
                   'United States': 'US',
                   'Viet Nam': 'Vietnam',
                   'Cabo Verde': 'Cape Verde',
                   'Brunei Darussalam': 'Brunei',
                   'Lao PDR': 'Laos',
                   'North Macedonia': 'Macedonia',
                   'Moldova Republic of': 'Moldova',
                   'Russian Federation': 'Russia',
                   'Eswatini': 'Swaziland',
                   "Côte d'Ivoire": 'Ivory Coast',
                   'Republic of the Congo': 'Congo',
                   'Democratic Republic of the Congo': 'Congo DR',
                   'Palestinian Territory': 'Palestine'}
    cont['Country'] = cont['Country'].replace(change_name.keys(), change_name.values())

    df4 = pd.merge(left=df4, right=cont, how='inner')

    for i in range(11):
        for j in range(11):
            df4['Covid_19_economic_exposure_index'] = df4[['Covid_19_economic_exposure_index']].replace(str(i)
                                                                    +','+str(j), float(str(i)+'.'+str(j)))

    df4 = df4[(df4['Covid_19_economic_exposure_index'] != 'x') & (df4['Covid_19_economic_exposure_index'] != "No data")]
    cont_list = ['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania']
    df4_cont_list = df4['Continent'].to_list()
    df4_econ_list = df4['Covid_19_economic_exposure_index'].to_list()
    avg_list = []
    for i in range(len(cont_list)):
        econ = 0
        count = 0
        for j in range(len(df4_cont_list)):
            if cont_list[i] == df4_cont_list[j]:
                econ += df4_econ_list[j]
                count += 1
        avg_econ = econ/count
        avg_list.append(avg_econ)

    df4_dict = {'Continent': cont_list, 'average_covid_19_Economic_exposure_index': avg_list}
    df4 = pd.DataFrame(df4_dict)
    df4 = df4.set_index('Continent')
    df4 = df4.sort_values(by='average_covid_19_Economic_exposure_index', ascending=True)




    #################################################

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    def cal_avg_invest(clas, country_list, df5):
        sum_invest = 0
        count = 0
        for i in country_list:
            if df5.loc[i, 'Income classification according to wb'] == clas:
                invest_val = float(df5.loc[i, 'Foreign direct investment'].replace(',', '.'))
                sum_invest += invest_val
                count += 1
        avg_invest = sum_invest / count
        return avg_invest

    def cal_avg_receive(clas, country_list, df5):
        sum_receive = 0
        count = 0
        for i in country_list:
            if df5.loc[i, 'Income classification according to wb'] == clas:
                receive_val = float(df5.loc[i, 'Net_oda_received_perc_of_gni'].replace(',', '.'))
                sum_receive += receive_val
                count += 1
        avg_receive = sum_receive / count
        return avg_receive


    df5 = df2.copy()
    class_list = ['HIC', 'MIC', 'LIC']
    df5 = df5.reset_index()



    df5 = df5[(df5['Foreign direct investment'] != 'x') & (df5['Foreign direct investment'] != 'No Data') &
              (df5['Net_oda_received_perc_of_gni'] != 'x') & (df5['Net_oda_received_perc_of_gni'] != 'No Data') &
              (df5['Net_oda_received_perc_of_gni'] != 'No data')]
    # df5 = df5[(df5['Net_ODA_received_perc_of_GNI'] != 'x') & (df5['Net_ODA_received_perc_of_GNI'] != 'No Data')]
    country_list = df5['Country'].to_list()
    df5 = df5.set_index('Country')
    avg_invest_list = [cal_avg_invest(class_list[0], country_list, df5),
                       cal_avg_invest(class_list[1], country_list, df5),
                       cal_avg_invest(class_list[2], country_list, df5)]
    avg_receive_list = [cal_avg_receive(class_list[0], country_list, df5),
                        cal_avg_receive(class_list[1], country_list, df5),
                        cal_avg_receive(class_list[2], country_list, df5)]

    data_dict = {'Income Class': class_list,
                 'Avg Foreign direct investment': avg_invest_list,
                 'Avg_ Net_ODA_received_perc_of_GNI': avg_receive_list}
    df5 = pd.DataFrame(data_dict)
    df5 = df5.set_index('Income Class')

    #################################################

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []
    #################################################
    # Your code goes here ...

    df6 = df2.copy()

    df6 = df6[df6['Income classification according to wb'] == 'LIC']
    df6 = df6['Cities']
    df6_dict = df6.to_dict()
    lic_countries = list(df6_dict.keys())
    lic_cities_info = list(df6_dict.values())
    low_cities_list = []
    popu_list = []
    for i in range(len(lic_countries)):
        lic_cities_info[i] = lic_cities_info[i].split("|||")
        # lic_cities_info2[i] = lic_cities_info2[i].split("|||")
        for j in range(len(lic_cities_info[i])):
            temp = lic_cities_info[i][j].split(',')
            temp1 = temp[4]
            city_info = temp[1]
            city_name = city_info.split(':')[1]
            temp2 = temp1.split(':')[1]
            if (temp2 != 'null') and (temp2 != 'null}'):
                if '}' in temp2:
                    temp2 = temp2.replace('}', '')
                popu_list.append(float(temp2))
                low_cities_list.append(city_name)
    for i in range(5):
        cities_lst.append(str(low_cities_list[popu_list.index(max(popu_list))]).replace('"', ''))

        drop_index = popu_list.index(max(popu_list))
        popu_list.pop(drop_index)
        low_cities_list.pop(drop_index)

    lst = cities_lst

    #################################################

    log("QUESTION 6", output_df=None, other=cities_lst)
    return lst


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    df7 = df2.copy()
    df7_dict = df7['Cities'].to_dict()
    counrty_list = list(df7_dict.keys())
    city_info_list = list(df7_dict.values())
    city_country_list = []
    for i in range(len(counrty_list)):
        city_info_list[i] = city_info_list[i].split("|||")
        for j in range(len(city_info_list[i])):
            city_info_list[i][j] = city_info_list[i][j].split(',')
            city_name = city_info_list[i][j][1].split(':')[1]
            city_name = city_name.replace('"', '')
            city_country_list.append([city_name, counrty_list[i]])

    uni_dict = {}
    for i in range(len(city_country_list)):
        if city_country_list[i][0] not in uni_dict.keys():
            uni_dict[city_country_list[i][0]] = [city_country_list[i][1]]
        else:
            uni_dict[city_country_list[i][0]].append(city_country_list[i][1])
    result_dict = {}
    for k, v in uni_dict.items():
        if len(v) >= 2:
            result_dict[k] = v
    uni_cities_list = list(result_dict.keys())
    uni_countries_list = list(result_dict.values())
    for i in range(len(uni_countries_list)):
        uni_countries_list[i] = list(set(uni_countries_list[i]))

    df_dict = {'City': uni_cities_list, 'Countries': uni_countries_list}
    df7 = pd.DataFrame(df_dict)
    df7 = df7.set_index('City')

    #################################################

    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    def cal_world_popu(world_countries_list, world_cities_info_list):
        popu_num_list = []
        for i in range(len(world_countries_list)):
            world_cities_info_list[i] = world_cities_info_list[i].split("|||")
            for j in range(len(world_cities_info_list[i])):
                temp = world_cities_info_list[i][j].split(',')
                temp1 = temp[4]
                temp2 = temp1.split(':')[1]
                if (temp2 != 'null') and (temp2 != 'null}'):
                    if '}' in temp2:
                        temp2 = temp2.replace('}', '')
                    popu_num_list.append(float(temp2))
        world_popu_sum = sum(popu_num_list)
        return world_popu_sum

    def cal_country_popu(sa_cities_info_list):
        popu_num_list = []
        for i in range(len(sa_cities_info_list)):
            temp = sa_cities_info_list[i].split(',')
            temp1 = temp[4]
            temp2 = temp1.split(':')[1]
            if (temp2 != 'null') and (temp2 != 'null}'):
                if '}' in temp2:
                    temp2 = temp2.replace('}', '')
                popu_num_list.append(float(temp2))
        country_popu_sum = sum(popu_num_list)
        return country_popu_sum

    df8 = df2.copy()
    df8 = df8.reset_index()
    df_continent_country = pd.read_csv(continents, encoding="ISO-8859-1")
    change_name = {'Korea DPR': 'North Korea',
                   'Korea, North': 'North Korea',
                   'Korea Republic of': 'South Korea',
                   'Korea, South': 'South Korea',
                   'United States of America': 'US',
                   'United States': 'US',
                   'Viet Nam': 'Vietnam',
                   'Cabo Verde': 'Cape Verde',
                   'Brunei Darussalam': 'Brunei',
                   'Lao PDR': 'Laos',
                   'North Macedonia': 'Macedonia',
                   'Moldova Republic of': 'Moldova',
                   'Russian Federation': 'Russia',
                   'Eswatini': 'Swaziland',
                   "Côte d'Ivoire": 'Ivory Coast',
                   'Republic of the Congo': 'Congo',
                   'Democratic Republic of the Congo': 'Congo DR',
                   'Palestinian Territory': 'Palestine'}
    df_continent_country['Country'] = df_continent_country['Country'].replace(change_name.keys(), change_name.values())
    temp_df = pd.merge(left=df8, right=df_continent_country, how='inner')
    temp_df = temp_df.set_index('Country')
    world_countries_df = temp_df['Cities']

    world_countries_dict = world_countries_df.to_dict()
    world_countries_list = list(world_countries_dict.keys())
    world_cities_info_list = list(world_countries_dict.values())
    world_popu_sum = cal_world_popu(world_countries_list, world_cities_info_list)

    sa_countries_df = temp_df[temp_df['Continent'] == 'South America']
    sa_countries_df = sa_countries_df['Cities']
    sa_countries_dict = sa_countries_df.to_dict()
    sa_countries_list = list(sa_countries_dict.keys())
    sa_cities_info_list = list(sa_countries_dict.values())
    sa_countries_popu_list = []
    for i in range(len(sa_countries_list)):
        sa_cities_info_list[i] = sa_cities_info_list[i].split("|||")
        country_popu_sum = cal_country_popu(sa_cities_info_list[i])
        sa_countries_popu_list.append(country_popu_sum)
    percent_list = []
    for i in range(len(sa_countries_list)):
        percent_list.append(sa_countries_popu_list[i]/world_popu_sum * 100)

    # print(world_popu_sum)
    # print(sa_countries_popu_list)
    plt.figure(figsize=(10, 10))
    plt.bar(range(len(sa_countries_list)), percent_list, width=0.5)
    plt.xticks(range(len(sa_countries_list)), sa_countries_list, rotation=60)
    plt.xlabel("Countries")
    plt.ylabel("percentage(%)")
    plt.title("Percentage of the world population living in each South American country")
    plt.grid(alpha=0.4)
    ###################
    # index = np.arange(len(sa_countries_list))
    # for a, b in zip(index, sa_countries_popu_list):
    #     plt.text(a, b, '%.2f'%b, ha='center', va='bottom', fontsize=15)

    #################################################

    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    def cal_avg(data_list):
        sum_data = 0
        count = 0
        for i in range(len(data_list)):
            if data_list[i] != 'x':
                data_list[i] = float(data_list[i].replace(',', '.'))
                sum_data += data_list[i]
                count += 1
        avg_result = float(sum_data/count)
        return avg_result

    df9 = df2.copy()
    df_HIC = df9[df9['Income classification according to wb'] == 'HIC']
    df_HIC = df_HIC[['Income classification according to wb',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import',
                     'Foreign direct investment, net inflows percent of gdp',
                     'Foreign direct investment']]
    list_HIC_1 = df_HIC['Covid_19_economic_exposure_index_ex_aid_and_fdi'].to_list()
    list_HIC_2 = df_HIC['Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import'].to_list()
    list_HIC_3 = df_HIC['Foreign direct investment, net inflows percent of gdp'].to_list()
    list_HIC_4 = df_HIC['Foreign direct investment'].to_list()


    df_LIC = df9[df9['Income classification according to wb'] == 'LIC']
    df_LIC = df_LIC[['Income classification according to wb',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import',
                     'Foreign direct investment, net inflows percent of gdp',
                     'Foreign direct investment']]
    list_LIC_1 = df_LIC['Covid_19_economic_exposure_index_ex_aid_and_fdi'].to_list()
    list_LIC_2 = df_LIC['Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import'].to_list()
    list_LIC_3 = df_LIC['Foreign direct investment, net inflows percent of gdp'].to_list()
    list_LIC_4 = df_LIC['Foreign direct investment'].to_list()

    df_MIC = df9[df9['Income classification according to wb'] == 'MIC']
    df_MIC = df_MIC[['Income classification according to wb',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi',
                     'Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import',
                     'Foreign direct investment, net inflows percent of gdp',
                     'Foreign direct investment']]
    list_MIC_1 = df_MIC['Covid_19_economic_exposure_index_ex_aid_and_fdi'].to_list()
    list_MIC_2 = df_MIC['Covid_19_economic_exposure_index_ex_aid_and_fdi_and_food_import'].to_list()
    list_MIC_3 = df_MIC['Foreign direct investment, net inflows percent of gdp'].to_list()
    list_MIC_4 = df_MIC['Foreign direct investment'].to_list()
    result_df = pd.DataFrame({
        'Income Class': ['HIC', 'MIC', 'LIC'],
        'Covid_19_Economic_exposure_index_Ex_aid_and_FDI':
            [cal_avg(list_HIC_1), cal_avg(list_MIC_1), cal_avg(list_LIC_1)],
        'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import':
            [cal_avg(list_HIC_2), cal_avg(list_MIC_2), cal_avg(list_LIC_2)],
        'Foreign direct investment, net inflows percent of GDP':
            [cal_avg(list_HIC_3), cal_avg(list_MIC_3), cal_avg(list_LIC_3)],
        'Foreign direct investment':
            [cal_avg(list_HIC_4), cal_avg(list_MIC_4), cal_avg(list_LIC_4)]
    })
    # plt.figure(figsize=(30, 15))
    result_df.plot.bar(x='Income Class', figsize=(15, 10))
    plt.grid(alpha=0.4)
    plt.legend(loc='upper left')
    #################################################

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    #################################################
    # Your code goes here ...

    def cal_country_popu(country_cities_info):
        country_cities_info_list = country_cities_info.split("|||")
        cities_popu_list = []
        for i in range(len(country_cities_info_list)):
            temp = country_cities_info_list[i].split(',')
            temp2 = temp[4].split(':')[1]
            if (temp2 != 'null') and (temp2 != 'null}'):
                if '}' in temp2:
                    temp2 = temp2.replace('}', '')
                cities_popu_list.append(float(temp2))
        country_popu_sum = sum(cities_popu_list)
        return country_popu_sum


    df10 = df2.copy()
    df10 = df10.reset_index()
    countries_continents_df = pd.read_csv(continents)
    change_name = {'Korea DPR': 'North Korea',
                   'Korea, North': 'North Korea',
                   'Korea Republic of': 'South Korea',
                   'Korea, South': 'South Korea',
                   'United States of America': 'US',
                   'United States': 'US',
                   'Viet Nam': 'Vietnam',
                   'Cabo Verde': 'Cape Verde',
                   'Brunei Darussalam': 'Brunei',
                   'Lao PDR': 'Laos',
                   'North Macedonia': 'Macedonia',
                   'Moldova Republic of': 'Moldova',
                   'Russian Federation': 'Russia',
                   'Eswatini': 'Swaziland',
                   "Côte d'Ivoire": 'Ivory Coast',
                   'Republic of the Congo': 'Congo',
                   'Democratic Republic of the Congo': 'Congo DR',
                   'Palestinian Territory': 'Palestine'}
    countries_continents_df['Country'] = countries_continents_df['Country'].replace(change_name.keys(), change_name.values())
    # copy_df = countries_continents_df

    df10 = pd.merge(left=df10, right=countries_continents_df, how='inner')
    df10 = df10[['Country', 'Continent', 'avg_latitude', 'avg_longitude', 'Cities']]
    avg_lon_list = df10['avg_longitude'].to_list()
    avg_lat_list = df10['avg_latitude'].to_list()
    world_cities_info_list = df10['Cities'].to_list()
    countries_list = df10['Country'].to_list()
    country_cont_list = df10['Continent'].to_list()
    cont_list = ['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Oceania']
    color_list = ['orange', 'black', 'blue', 'red', 'green', 'brown']
    countries_popu_list = []
    for i in range(len(countries_list)):
        country_popu_num = cal_country_popu(world_cities_info_list[i])
        countries_popu_list.append(country_popu_num)

    plt.figure()

    for i in range(len(country_cont_list)):
        for j in range(len(cont_list)):
            if country_cont_list[i] == cont_list[j]:
                colour = color_list[j]
        plt.scatter(x=avg_lon_list[i], y=avg_lat_list[i], s=countries_popu_list[i]/500000, color=colour, alpha=0.7)
    for i in range(len(cont_list)):
        plt.scatter(x=[], y=[], color=color_list[i], label=cont_list[i])

    plt.grid(alpha=0.4)
    plt.xlabel("avg_longitude")
    plt.ylabel("avg_latitude")
    plt.legend(loc='lower left', fontsize='x-small')
    #################################################

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")