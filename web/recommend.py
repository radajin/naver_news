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

def analytics_comments(comments):
    
    category_dict = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0}
    classification_model = pickle.load(open("./models/classification_model.plk", "rb"))
    category_list = []

    for comment in comments:
        category = str(classification_model.predict([comment])[0])
        category_list.append(category)
        category_dict[str(category)] += 1

    max_category = max(category_dict.items(), key=operator.itemgetter(1))[0]

    return category_dict, max_category, category_list

def category_recommend(category):
	
	article_df, comment_df = load_datas()

	return article_df[article_df["category"] == int(category)].sort_values(by="comment", ascending=False)	


def recommend(userId):

	def remove_duplicate(list1, list2):
		for idx in list2:
			list1 = [x for x in list1 if x != idx]
		return list1

    # load model & aritcle, comment dataframe
	recommend_model = pickle.load(open("./models/recommend_model.plk", "rb"))
	article_df, comment_df = load_datas()

	# set data from model
	unique_user = recommend_model['unique_user']
	article_list = recommend_model['article_list']
	datas = recommend_model['datas']
	predict = recommend_model['predict']

	# find user index	
	idx = list(unique_user).index(userId)

	# set recommend article & recommend predict point
	recomend_article = article_list[datas[idx, :] == 0]
	recomend_predict = predict[idx, :][datas[idx, :] == 0]
	recomend_article = recomend_article[recomend_predict > 0]
	recomend_predict = recomend_predict[recomend_predict > 0]

	# set return datas
	recommend_article_list = []
	comments = list(comment_df[comment_df["userIdNo"] == userId]["contents"])
	category_dict, max_category, category_list = analytics_comments(comments)

	# article list
	aritcle_list = list(category_recommend(max_category)["newsid"])

	# comment list
	tmp_df = comment_df[comment_df["userIdNo"] == userId]
	comment_list = list(set(tmp_df["aid"]))

	# remove duplication
	aritcle_list = remove_duplicate(aritcle_list, comment_list)[:5]

	if len(recomend_article) != 0:

		# recommend article sorting
		result_list = []

		for i in range(len(recomend_article)):
			result_list.append((recomend_article[i], recomend_predict[i]))

		sorted_recommend_article = sorted(result_list, key=lambda tup: tup[1])
		recommend_aritcle_list, dist_list = zip(*sorted_recommend_article)
		recommend_aritcle_list = recommend_aritcle_list[::-1]
		
		# remove duplicate
		aritcle_list = remove_duplicate(aritcle_list, recommend_aritcle_list)

		# concat recommend_article + category_recommend_article
		aritcle_list = list(recommend_aritcle_list) + list(article_list)
		aritcle_list = aritcle_list[:5]
	else:
		print("No Recomend")
		
	# set result recomend article list
	for aritcle in aritcle_list:
		article = article_df[article_df["newsid"] == int(aritcle)]
		recommend_dict = {
			'newspaper': article['newspaper'].values[0],
			'title': article['title'].values[0],
			'link': article['link'].values[0],
			'content': article['content'].values[0],
		}
		recommend_article_list.append(recommend_dict)
         
	return recommend_article_list, comments, category_dict, max_category, category_list

def userList():
	recommend_model = pickle.load(open("./models/recommend_model.plk", "rb"))

	# set data from model
	return list(recommend_model['unique_user'])

	# userList = ['28qA1', '7G80r', '85fbU', '3EQjn', 'Iqis', 'jE62', '5UM3g', '6j7iu', '3Bpiw', '6ij6t']
	# return userList

def mae_mean():
	def mae(data, predict):
		delta = data[data > 0] - predict[data > 0]
		return np.absolute(delta).sum()/len(delta)

	recommend_model = pickle.load(open("./models/recommend_model.plk", "rb"))
	
	# set data from model
	datas = recommend_model['datas']
	predict = recommend_model['predict']
	unique_user = recommend_model['unique_user']
	article_list = recommend_model['article_list']

	mae_list = []

	for idx in range(len(datas)):
		result_mae = mae(datas[idx,:], predict[idx,:])
		mae_list.append(result_mae)

	return np.array(mae_list).mean(), len(unique_user), len(article_list)


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
		recommend_article_list, comments, category_dict, max_category, category_list = recommend(userId)

		result = {
			'recommend_article_list': recommend_article_list, 
			'comments':comments, 
			'category_dict':category_dict, 
			'max_category':max_category,
			'category_list':category_list,
			'status_code': 200, 
		}	

	elif command == "userList":

		result = {
			'user': userList(),
			'status_code': 200, 
		}

	elif command == "evaluation":
		mae, user_num, article_num = mae_mean()
		result = {
			'mae_mean' : mae,
			'article': article_num,
			'user': user_num,
			'status_code': 200, 
		}

	return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80, debug=True)




