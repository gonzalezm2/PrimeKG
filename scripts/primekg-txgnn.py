#directions for PrimeKG-TXGNN run
#ssh into cluster

bsub -M 64000 -Is bash

module add anaconda/3

#setup 
mkdir PrimeKG-TXGNN
cd PrimeKG-TXGNN/

conda create --name txgnn_env python=3.10

conda activate txgnn_env

cd /home/migonz/PrimeKG-TXGNN

#setup
#conda install pandas numpy scipy matplotlib scikit-learn networkx -y
#pip install pykeen pyvis
#pip install dgl==1.1.2
#pip install git+https://github.com/mims-harvard/TXGNN.git


ipython

from txgnn import TxData, TxGNN, TxEval
import torch
import pandas as pd
import matplotlib.pyplot as plt

nodes = pd.read_csv("/home/migonz/PrimeKG-TXGNN/data/node.csv")
edges = pd.read_csv("/home/migonz/PrimeKG-TXGNN/data/edges.csv")

#just looking at data
print("Nodes shape:", nodes.shape)
print("Edges shape:", edges.shape)
print("Node columns:", nodes.columns.tolist())
print("Edge columns:", edges.columns.tolist())
print("First few nodes:\n", nodes.head())
print("First few edges:\n", edges.head())
print("Relation types in KG:", edges['relation'].unique())

# Download/load knowledge graph dataset
TxData_primekg = TxData(data_folder_path = './data')


##supported splits are 'random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', or 'full_graph'or this TxData_primekg.prepare_split(split = 'disease_eval', disease_eval_idx = 'XX')
TxData_primekg.prepare_split(split = 'full_graph', seed = 42)
TxData_primekg.prepare_split(split='disease_eval', disease_eval_idx=3481)
TxData_primekg.prepare_split(split = 'random', seed = 42)

#only works when you split on complex_disease or other criteria
TxGNN_model = TxGNN(data = TxData_primekg, 
              weight_bias_track = False,
              proj_name = 'TxGNN', # wandb project name
              exp_name = 'TxGNN', # wandb experiment name
              device = torch.device('cpu') # define your cuda device
              )


TxGNN_model.model_initialize(n_hid = 100, # number of hidden dimensions
                      n_inp = 100, # number of input dimensions
                      n_out = 100, # number of output dimensions
                      proto = True, # whether to use metric learning module
                      proto_num = 3, # number of similar diseases to retrieve for augmentation
                      attention = False, # use attention layer (if use graph XAI, we turn this to false)
                      sim_measure = 'all_nodes_profile', # disease signature, choose from ['all_nodes_profile', 'protein_profile', 'protein_random_walk']
                      agg_measure = 'rarity', # how to aggregate sim disease emb with target disease emb, choose from ['rarity', 'avg']
                      num_walks = 200, # for protein_random_walk sim_measure, define number of sampled walks
                      path_length = 2 # for protein_random_walk sim_measure, define path length
                      )

#load pretrained model, download from here: https://drive.google.com/file/d/1fxTFkjo2jvmz9k6vesDbCeucQjGRojLj/view
TxGNN_model.load_pretrained('./TxGNNExplorer')

#can also run these, but not necessary if loading the pretrained model 
TxGNN_model.pretrain(n_epoch = 2, 
               learning_rate = 1e-3,
               batch_size = 1024, 
               train_print_per_n = 20)


#can also run this if building own model from scratch
TxGNN_model.finetune(n_epoch = 500, 
               learning_rate = 5e-4,
               train_print_per_n = 5,
               valid_per_n = 20)


#TxEval_random = TxEval(model = TxGNN_model)

TxEval_all = TxEval(model = TxGNN_model)

#indvidual disease results
result_castle = TxEval_all.eval_disease_centric(disease_idxs = [3481], 
                                     relation = 'indication', 
                                     save_result = False)


#derive internal node ids from pretrained model
mapping = TxData_primekg.retrieve_id_mapping()
idx2id_disease = mapping['idx2id_disease'] 
idx2id_drug = mapping['idx2id_drug'] 
id2name_disease = mapping['id2name_disease'] 
id2name_drug = mapping['id2name_drug']
id2idx_disease = {j:i for i,j in idx2id_disease.items()}


##This will give a spreadsheet of all internal disease nodes to call in results
#Invert idx2id_disease so you get: {id: idx}
id2idx_disease = {v: k for k, v in idx2id_disease.items()}

#Create list of tuples: (index, ID, name)
rows = []
for disease_id, disease_name in id2name_disease.items():
    idx = id2idx_disease.get(disease_id)
    rows.append((idx, disease_id, disease_name))

#Build DataFrame
df_diseases = pd.DataFrame(rows, columns=['Index', 'ID', 'Name'])

#ort by index
df_diseases = df_diseases.sort_values(by='Index')

print(df_diseases.head())
df_diseases.to_csv("all_diseases_idx_id_name.csv", index=False)



#define internal disease nodes so you can output in results
disease_idxs = [
    16427, 2570, 3028, 4536, 15428, 486, 1036, 17047, 15651, 1009, 15385, 2285, 16779,
    15665, 15277, 15576, 14574, 15093, 275, 14975, 14665, 14624, 15896, 15632, 16564,
    15519, 15264, 1322, 1254, 16804, 7189, 16347, 15103, 1616, 6422, 7082, 12946, 1225,
    3388, 6121, 5015, 6616, 6572, 5561, 3481, 14097, 5308, 7073, 5217, 3231, 6817, 3883,
    6303, 6584, 6410, 6451, 6592, 13412, 6586, 6263, 3823, 6373, 6408, 6407, 12725, 5764,
    5005, 4024, 6386, 3521, 5961, 7936, 9285, 7998, 5554, 14401, 12772, 8519, 12896,
    12651, 12726, 12709, 14, 10116, 12948, 12751, 3912, 3988, 6974, 1479, 14098, 12900,
    6036, 12095, 6760, 6893, 4163, 6885, 13131, 14303, 6765, 1321, 4222, 518, 1396, 3155,
    6734, 7354, 7858, 8370, 8371, 8518, 8893, 10102, 10628, 12083
]

autoinflam_disease_idxs = [5592, 5255, 14782, 1435, 5877, 12498, 15783, 6185, 9030, 3468, 5767, 16910, 2736]

#run this to get results for diseases of interest
result_auto = TxEval_all.eval_disease_centric(
    disease_idxs = disease_idxs,
    relation = 'indication',
    save_result = False
)

#run this to get results for diseases of interest
result_autoinflam = TxEval_all.eval_disease_centric(
    disease_idxs = autoinflam_disease_idxs,
    relation = 'indication',
    save_result = False
)

#Save DataFrame to pickle
result_auto.to_pickle("result_auto.pkl")
result_auto.to_pickle("result_autoinflam.pkl")

#load into python
result_auto = pd.read_pickle("result_auto.pkl")
result_autoinflam = pd.read_pickle("result_autoinflam.pkl")


#Reverse the mapping so we can go from name → DrugBank ID
name2id_drug = {v: k for k, v in id2name_drug.items()}

#Collect results
all_rows = []

for i in range(len(result_auto)):
    row = result_auto.iloc[i]
    disease_id = row["ID"]
    disease_name = row["Name"]
    predictions = row["Prediction"]
    fda_hits = row["Hits@100"]

    if not isinstance(predictions, dict):
        continue

    for drug_id, score in predictions.items():
        drug_name = id2name_drug.get(drug_id, "Unknown")
        fda_approved = drug_name in fda_hits if isinstance(fda_hits, list) else False
        all_rows.append([disease_id, disease_name, drug_name, drug_id, score, fda_approved])

#Create final DataFrame
df_all = pd.DataFrame(all_rows, columns=[
    "Disease ID", "Disease Name", "Drug Name", "DrugBank ID", "Prediction Score", "FDA Approved"
])

#Preview
print(df_all.head())

##normalize scores by disease
1- (rank/total number of drugs)

#Rank drugs by prediction score within each disease
df_all["Rank"] = df_all.groupby("Disease ID")["Prediction Score"].rank(ascending=False, method="first")

#Total number of drugs for each disease
df_all["Total Drugs"] = df_all.groupby("Disease ID")["Drug Name"].transform("count")

#Normalized score
df_all["Normalized Score"] = 1 - (df_all["Rank"] / df_all["Total Drugs"])

#Save to CSV
df_all.to_csv("disease_drug_predictions_normalized.csv", index=False)



##same as above, but with autoinflammatory diseases
#Reverse the mapping so we can go from name → DrugBank ID
name2id_drug = {v: k for k, v in id2name_drug.items()}

#Collect results
all_rows = []

for i in range(len(result_autoinflam)):
    row = result_autoinflam.iloc[i]
    disease_id = row["ID"]
    disease_name = row["Name"]
    predictions = row["Prediction"]
    fda_hits = row["Hits@100"]

    if not isinstance(predictions, dict):
        continue

    for drug_id, score in predictions.items():
        drug_name = id2name_drug.get(drug_id, "Unknown")
        fda_approved = drug_name in fda_hits if isinstance(fda_hits, list) else False
        all_rows.append([disease_id, disease_name, drug_name, drug_id, score, fda_approved])

#Create final DataFrame
df_all = pd.DataFrame(all_rows, columns=[
    "Disease ID", "Disease Name", "Drug Name", "DrugBank ID", "Prediction Score", "FDA Approved"
])

print(df_all.head())

#normalize scores by disease
1- (rank/total number of drugs)

#Rank drugs by prediction score within each disease
df_all["Rank"] = df_all.groupby("Disease ID")["Prediction Score"].rank(ascending=False, method="first")

#Total number of drugs for each disease
df_all["Total Drugs"] = df_all.groupby("Disease ID")["Drug Name"].transform("count")

#Normalized score
df_all["Normalized Score"] = 1 - (df_all["Rank"] / df_all["Total Drugs"])

#Save to CSV
df_all.to_csv("autoinflam_disease_drug_predictions_normalized.csv", index=False)
