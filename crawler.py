import requests
from pyquery import PyQuery as pq
import numpy as np
from member import Member as Member
from sklearn.cluster import KMeans
import sklearn
import pandas as pd
from functools import reduce
import jupyter

def main():
	votes_data = {}
	
	divisionList = [655,656,657,658,659,660,661,662]
	voteType = ['ayes', 'noes', 'notrecorded']
	
	for x in divisionList:
			members = []
		#for vType in voteType:
			#dom = goToPage('https://commonsvotes.digiminster.com/Divisions/Details/'+str(x)+'#'+vType)
			dom = goToPage('https://commonsvotes.digiminster.com/Divisions/Details/'+str(x))
			print('Crawling votes on division '+str(x))
			#activeTab = dom('.tab-pane active')
			#elements = pq(activeTab).children('.header') #care are tab-pane active !! si dupa ce aflu doar elementele astea, trb sa aflu care e ID ul ayesList sau noesList sau pzdm
			elements = dom('.details-card-inner')
			print(len(elements))
			#trebuie sa iau si numele la division -- trebuie sa iau first header din pag si sa caut children cu .title
			divisionName = pq(dom('.title')[0]).text()
			#print(divisionName)
			for item in elements:
				member_name = pq(item).children('.header').children('.title').text().replace(",", "")
				member_constituency = pq(item).children('.header').children('.constituency').text()
				member_party = pq(item).children('.header').children('.party').text()
				voteList = pq(item).parents('.tab-pane').attr('id')
				if voteList == 'ayesList':
					member_vote = 1
				elif voteList == 'noesList':
					member_vote = -1
				else: 
					member_vote == 0
				member = Member(member_name, member_constituency, member_party, member_vote)
				members.append(member)
			#print(members)
			votes_data[divisionName] = members
			
	
	print(votes_data)
	#pandas data frame 
	#df = pd.DataFrame(data=votes_data)
	df = pd.DataFrame.from_dict(votes_data, orient='index')
	df.transpose()
	#df = reduce(lambda x,y: x.join(y), df)
	df.reset_index(inplace=True)
	df.head()

def goToPage(url):
    response = requests.get(url)
    html = response.text
    dom = pq(html)
    return dom


if __name__ == '__main__':
	main()