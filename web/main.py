from flask import Flask, render_template, jsonify, session, request

import pickle, operator, itertools
import numpy as np
import scipy as sp
from scipy import spatial
import pandas as pd

app = Flask(__name__)

def load_datas(date="2016-06-01"):

	# load article df
	file = open("../data/article_" + date + ".plk", 'rb')
	article_df = pickle.load(file)
	article_df = article_df[np.invert(article_df.duplicated(subset="newsid"))] # remove duplication
	article_df = article_df[article_df["comment"] > 500]
	file.close()
    
	# load commend df
	file = open("../data/comment_" + date + ".plk", 'rb')
	comment_df = pickle.load(file)
	comment_df = comment_df[(comment_df["good"] > 0) & (comment_df["bad"] > 0)].reset_index(drop=True) # remove good:0, bad:0 
	comment_df = comment_df[comment_df["userIdNo"].str.len() < 10] # remove userIdNo > 10
	comment_df["aid"] = comment_df["aid"].apply(lambda aid: int(aid)) # change aid data type to int
	file.close()

	return article_df, comment_df

def make_datas(article_df, comment_df):
    
	# make zeros datas
	unique_user = comment_df["userIdNo"].unique()
	article_list = np.array(article_df["newsid"])
	datas = np.zeros([len(unique_user), len(article_list)])
    
	for idx, row in comment_df.iterrows():
        
		userIdNo = row["userIdNo"]
		aid = row["aid"]
        
        # continue when no aid in article_list
		if aid not in article_list: 
			continue
            
		# fill values
		row_idx = int(np.where(unique_user==userIdNo)[0])
		aid_idx = int(np.where(article_list==int(aid))[0])
		value = int(datas[row_idx, aid_idx:aid_idx+1])
		datas[row_idx, aid_idx:aid_idx+1] = value + 1

    # reduce datas
	unique_user = unique_user[datas.sum(axis=1) > 10]
	datas = datas[datas.sum(axis=1) > 10]
	article_list = article_list[datas.sum(axis=0) > 0]
	datas = datas[:,datas.sum(axis=0) > 0]

	return datas, unique_user, article_list

def make_predict(datas):

	def predict_vector(datas, target_idx):
    
		dists = [ 
			(idx, spatial.distance.cosine(datas[target_idx,:], data))
			for idx, data in enumerate(datas) 
			if target_idx != idx
		]
	     
		dist_list = sorted(dists, key=lambda tup: tup[1])
		dist_index, dist_value = zip(*dist_list)
	    
		# remove value 1 sample
		dist_index = np.array(dist_index)[np.array(dist_value) > 0][:5]
		dist_value = np.array(dist_value)[np.array(dist_value) > 0][:5]
	    
		return datas[dist_index,:].mean(axis=0)
	    
	predict_vectors = []

	for idx, data in enumerate(datas):
		predict_vectors.append( predict_vector(datas, idx) )

	return np.array(predict_vectors)

def mae_mean(datas, predict):
	def mae(data, predict):
		delta = data[data > 0] - predict[data > 0]
		return np.absolute(delta).sum()/len(delta)

	mae_list = []

	for idx in range(len(datas)):
		result_mae = mae(datas[idx,:], predict[idx,:])
		mae_list.append(result_mae)
	return np.array(mae_list).mean()

def make_model(date="2016-06-01"):

	article_df, comment_df = load_datas(date)
	datas, unique_user, article_list = make_datas(article_df, comment_df)
	predict = make_predict(datas)

	save_dict = {
		'datas':datas, 
		'unique_user':unique_user, 
		'article_list':article_list, 
		'predict':predict,
	}

	pickle.dump(save_dict, open("./models/recommend_model.plk", "wb"))

	mae = mae_mean(datas, predict)
	comment_mean = datas.sum(axis=1).mean()

	return {
		'datas': len(datas), 
		'unique_user': len(unique_user), 
		'article_list': len(article_list), 
		'predict': len(predict),
		'mae': mae,
		'comment_mean': comment_mean,
		'user_list': list(unique_user)[:10],
	}

def analytics_comments(comments):
    category_dict = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0}
    classification_model = pickle.load(open("./models/classification_model.plk", "rb"))
    for comment in comments:
        category = str(classification_model.predict([comment])[0])
        category_dict[str(category)] += 1
    max_category = max(category_dict.items(), key=operator.itemgetter(1))[0]
    return category_dict, max_category 

def category_recommend(category):
	article_df, comment_df = load_datas()
	return article_df[article_df["category"] == int(category)].sort_values(by="comment", ascending=False)	

def recommend(userId):
    
	model = pickle.load(open("./models/recommend_model.plk", "rb"))

	article_df, comment_df = load_datas()

	idx = list(model['unique_user']).index(userId)
	article_list = model['article_list']
	datas = model['datas']
	predict = model['predict']

	recomend_article = article_list[datas[idx, :] == 0]
	recomend_predict = predict[idx, :][datas[idx, :] == 0]

	recomend_article = recomend_article[recomend_predict > 0]
	recomend_predict = recomend_predict[recomend_predict > 0]
	    
	result_list = []

	for i in range(len(recomend_article)):
		result_list.append((recomend_article[i], recomend_predict[i]))
    
	sorted_recommend_article = sorted(result_list, key=lambda tup: tup[1])
	
	recommend_article_list = []
	comments = list(comment_df[comment_df["userIdNo"] == userId]["contents"])

	category_dict, max_category = analytics_comments(comments)

	aritcle_list = []

	# no recommand article
	if len(sorted_recommend_article) == 0:
		print("no recommand article")
		aritcle_list = list(category_recommend(max_category)["newsid"])[:5]

	else:
		aritcle_list, dist_list = zip(*sorted_recommend_article)
		aritcle_list = aritcle_list[::-1]

	for aritcle in aritcle_list:
		article = article_df[article_df["newsid"] == int(aritcle)]
		recommend_dict = {
			'newspaper': article['newspaper'].values[0],
			'title': article['title'].values[0],
			'link': article['link'].values[0],
			'content': article['content'].values[0],
		}
		recommend_article_list.append(recommend_dict)
         
	return recommend_article_list, comments, category_dict, max_category


# HTML webpage
@app.route('/')
def user():
    return render_template('index.html')

# retruns a piece of data in JSON format
@app.route('/api/<command>', methods=['GET', 'POST'])
def api(command):

	result = {}

	# recommend
	if command == "recommend":
		userId = request.args.get('userId', '')
		recommend_article_list, comments, category_dict, max_category = recommend(userId)
		result = {
			'recommend_article_list': recommend_article_list, 
			'comments':comments, 
			'category_dict':category_dict, 
			'max_category':max_category,
			'status_code': 200, 
		}	

	elif command == "makeRecommendModel":
		result = make_model()
		result['status_code'] = 200

	return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80, debug=True)











