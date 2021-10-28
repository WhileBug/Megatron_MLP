column_test_time_list = [{'test size': 512, 'device number': 0, 'phase 1 forward time': 2.4504131740993922e-05, 'phase 2 forward time': 0.00010699696011013455, 'phase 3 forward time': 0.0005810790591769748, 'backward time': 0.00045757823520236544, 'round number': 10}, {'test size': 1024, 'device number': 0, 'phase 1 forward time': 2.3788876003689235e-05, 'phase 2 forward time': 0.00020954344007703994, 'phase 3 forward time': 0.0006158881717258029, 'backward time': 0.0004811551835801866, 'round number': 10}, {'test size': 2048, 'device number': 0, 'phase 1 forward time': 3.843837314181858e-05, 'phase 2 forward time': 0.0009164280361599392, 'phase 3 forward time': 0.002171860800849067, 'backward time': 0.0011986096700032551, 'round number': 10}, {'test size': 4096, 'device number': 0, 'phase 1 forward time': 4.551145765516493e-05, 'phase 2 forward time': 0.005895641114976671, 'phase 3 forward time': 0.009461402893066406, 'backward time': 0.006317880418565538, 'round number': 10}, {'test size': 8192, 'device number': 0, 'phase 1 forward time': 5.1869286431206594e-05, 'phase 2 forward time': 0.03649852010938856, 'phase 3 forward time': 0.03093507554796007, 'backward time': 0.0386911498175727, 'round number': 10}, {'test size': 16384, 'device number': 0, 'phase 1 forward time': 5.5895911322699656e-05, 'phase 2 forward time': 0.3003832499186198, 'phase 3 forward time': 0.12642078929477268, 'backward time': 0.29727154307895237, 'round number': 10}]
row_test_time_list = [{'test size': 512, 'device number': 0, 'phase 1 forward time': 0.00011391109890407986, 'phase 2 forward time': 9.504954020182292e-05, 'phase 3 forward time': 0.00030154652065700956, 'backward time': 0.0006233056386311849, 'round number': 10}, {'test size': 1024, 'device number': 0, 'phase 1 forward time': 0.0001113944583468967, 'phase 2 forward time': 0.00019425816006130644, 'phase 3 forward time': 0.0010309219360351562, 'backward time': 0.0007563961876763238, 'round number': 10}, {'test size': 2048, 'device number': 0, 'phase 1 forward time': 0.00018599298265245225, 'phase 2 forward time': 0.00124359130859375, 'phase 3 forward time': 0.003930330276489258, 'backward time': 0.0013493167029486762, 'round number': 10}, {'test size': 4096, 'device number': 0, 'phase 1 forward time': 0.0002914004855685764, 'phase 2 forward time': 0.005845917595757378, 'phase 3 forward time': 0.01488147841559516, 'backward time': 0.006530417336357964, 'round number': 10}, {'test size': 8192, 'device number': 0, 'phase 1 forward time': 0.0006251335144042969, 'phase 2 forward time': 0.03673108418782552, 'phase 3 forward time': 0.0559746954176161, 'backward time': 0.038853194978502065, 'round number': 10}, {'test size': 16384, 'device number': 0, 'phase 1 forward time': 0.0016002125210232204, 'phase 2 forward time': 0.30155600441826713, 'phase 3 forward time': 0.22712948587205675, 'backward time': 0.2963417106204563, 'round number': 10}]
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def time_test_draw(time_list, savedir):
    time_dataframe = pd.DataFrame(time_list)
    print(time_dataframe)
    plt.figure(dpi=150)
    plt.clf()
    sns.barplot(x='test size',y='phase 1 forward time', data=time_dataframe)
    plt.savefig(savedir+"/phase 1 forward time.jpg")
    plt.clf()
    sns.barplot(x='test size',y='phase 2 forward time', data=time_dataframe)
    plt.savefig(savedir+"/phase 2 forward time.jpg")
    plt.clf()
    sns.barplot(x='test size',y='phase 3 forward time', data=time_dataframe)
    plt.savefig(savedir+"/phase 3 forward time.jpg")
    plt.clf()
    sns.barplot(x='test size',y='backward time', data=time_dataframe)
    plt.savefig(savedir+"/backward time.jpg")

time_test_draw(column_test_time_list,"column")
time_test_draw(row_test_time_list,"row")