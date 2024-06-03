import oracleVectorDB as ovs
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

file1 = open("./summary_animals.csv", "r")



from langchain_community.embeddings import OCIGenAIEmbeddings
auth_type="INSTANCE_PRINCIPAL"
compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"
GenAIEndpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

embedding_function=OCIGenAIEmbeddings(model_id="cohere.embed-multilingual-v3.0",
                                      service_endpoint=GenAIEndpoint,
                                      compartment_id=compartment_id,
                                      auth_type=auth_type)
# you can use oss embeddings
# embedding_function = HuggingFaceEmbeddings(
#     model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
tableName = 'ericanimal527'
ORACLE_AI_VECTOR_CONNECTION_STRING = "user/password@123.1.65.12:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"

oo = ovs.OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                        table=tableName)


def insert(oo):
    for line in file1.readlines():
        print(line)
        zy = line.split(',')
        path = zy[0]
        breed_file = zy[1]
        env_file = zy[2]
        action_file = zy[3]
        with open(breed_file.replace("\n", ""), "r") as f:
            breed = f.read()
            breed_v = embedding_function.embed_query(breed)
        with open(env_file.replace("\n", ""), "r") as f:
            env = f.read()
            env_v = embedding_function.embed_query(env)
        with open(action_file.replace("\n", ""), "r") as f:
            action = f.read()
            action_v = embedding_function.embed_query(action)

        oo.add_embedding(tableName, path, breed, breed_v,
                         env, env_v, action, action_v)


def searchVecs(query, typeSearch, k):
    vecs = embedding_function.embed_query(query)
    res = oo.similarity_search(
        embedding=vecs, table=tableName, typeSearch=typeSearch ,k=k)

    return res


if __name__ == '__main__':

    insert(oo)

