import os
from nl2cm.embed import get_embeddings


if __name__ == "__main__":
    data_path = "datasets/embeddings-dfs"
    
    print("Getting BPMN embeddings...")
    bpmn_nl_cm_cols = ['NL_Serialization_Emb_openai', 'CM_Serialization_Emb_openai']
    NLT_EMB, CM_EMB = get_embeddings(os.path.join(data_path, "bpmn"), bpmn_nl_cm_cols)
    print("Size of NLT_EMB: ", NLT_EMB.shape)
    print("Size of CM_EMB: ", CM_EMB.shape)
    
    print("Getting OntoUML embeddings...")
    ontouml_nl_cm_cols = ['NL_Serialization_Emb', 'CM_Serialization_Emb']
    NLT_EMB, CM_EMB = get_embeddings(os.path.join(data_path, "ontouml"), ontouml_nl_cm_cols)
    print("Size of NLT_EMB: ", NLT_EMB.shape)
    print("Size of CM_EMB: ", CM_EMB.shape)
    
    print("Getting Archimate embeddings...")
    archimate_nl_cm_cols = ['NL_Serialization_Emb', 'CM_Serialization_Emb']
    NLT_EMB, CM_EMB = get_embeddings(os.path.join(data_path, "archimate"), archimate_nl_cm_cols)
    print("Size of NLT_EMB: ", NLT_EMB.shape)
    print("Size of CM_EMB: ", CM_EMB.shape)
    