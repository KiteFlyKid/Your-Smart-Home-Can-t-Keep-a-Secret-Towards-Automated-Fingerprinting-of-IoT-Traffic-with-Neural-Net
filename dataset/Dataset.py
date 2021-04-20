import pandas as pd
import  os
import  numpy as np
import pickle

window_size=20 #40,100
def get_data(data_dir,label):
    X=[]
    df = pd.read_csv(data_dir)
    if df.shape[0]>5000:
        df=df.sample(5000,replace=False)
    for i in range(df.shape[0]-window_size):
        item=df.iloc[i:i + window_size, :].values.tolist()

        X.append((item,int(label)))
    return X

if __name__=='__main__':
    data_dir = 'processing_raw_data/final_output/'
    # total_df=pd.DataFrame()
    # for root, dirs, fnames in os.walk(data_dir):
    #         for fname in fnames:
    #             print('start reading {}'.format(fname))
    #             df=pd.read_csv(os.path.join(root, fname))
    #             if df.shape[0] > 5000:
    #                 df = df.sample(5000, replace=False)
    #             if len(total_df)==0:
    #                 total_df=df
    #             else:
    #                 total_df=pd.concat([total_df,df],axis=1)
    # final_df=total_df.reset_index(drop=True)
    # final_df=pd.get_dummies(total_df,columns=['dport'])
    # final_df.to_pickle('alltraffic.pkl')

    dataset = []
    label_dict={}
    for root, dirs, fnames in os.walk(data_dir):
        print(fnames)
        for fname in fnames:
            label_dict[fname]=len(label_dict)
            print('start reading {}'.format(fname))
            data = get_data(os.path.join(root, fname),label_dict[fname])
            dataset.extend(data)
            np.random.shuffle(dataset)
    with open('Dataset-Ind-train-{}.pkl'.format(window_size),'wb') as f:
            pickle.dump(dataset[:int(len(dataset)*0.7)],f)
    with open('Dataset-Ind-valid-{}.pkl'.format(window_size),'wb') as f:
            pickle.dump(dataset[int(len(dataset)*0.7):int(len(dataset) * 0.8)],f)
    with open('Dataset-Ind-test-{}.pkl'.format(window_size),'wb') as f:
            pickle.dump(dataset[int(len(dataset) * 0.8):],f)




