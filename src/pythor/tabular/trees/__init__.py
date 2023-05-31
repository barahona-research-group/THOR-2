"""
PyTorch 



cgbst.py
xgbst.py



Recommened Objectives for CatBoost 

    QueryRMSE
    

Not Recommened Objectives for CatBoost

    YetiRank
    PairLogit:max_pairs=10000 (Does not perform that well, also setting a magic number is not good)


Recommened Objetives for XGBoost 
    reg:squarederror

Not Recommended Objectives for XGBoost 

    rank:pairwise
    rank:ndcg


"""
