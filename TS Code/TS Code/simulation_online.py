import pandas as pd
from policies_offline import thompson_sampling
import urllib.request
import urllib.parse
import urllib.error
import json
import multiprocessing
from simulate_and_plot import plot_beta_dist, plot_assignment_prob, plot_success_failure
from time import sleep
import socket

def open_url_to_dict(url: str, data: bytes, method: str, header: dict, timeout=None):
	req = urllib.request.Request(url, data=data, headers=header, method=method)
	if timeout != None:
		response = urllib.request.urlopen(req, timeout=timeout)
	else:
		response = urllib.request.urlopen(req)
	return json.loads(response.read().decode())

def get_arm(mooclet: int, user_id: str, token: str, timeout=None):
	header = {'Authorization': 'Token ' + token}
	url = 'https://celery.mooclet.com/engine/api/v1/mooclet/{}/run?user_id={}'.format(mooclet, user_id)
	return open_url_to_dict(url, None, 'GET', header, timeout)

def get_arm_parallel(mooclet: int, user_id: str, token: str, versions: list, sim_count: int, wait_time: float, 
	q: multiprocessing.Queue):
	sim_result = {'probability_{}'.format(version): 0 for version in versions}
	for _ in range(sim_count):
		arm = None
		while arm == None:
			try:
				arm = get_arm(mooclet, 'learner', token)
				sim_result['probability_{}'.format(arm['id'])] += 1
			except (urllib.error.HTTPError, urllib.error.URLError) as e:
				print(str(e))
				sleep(wait_time)
			except Exception as e:
				print(str(e))
				raise e
	q.put(sim_result)

def send_reward(variable: str, value: float, user_id: str, mooclet: int, version: int, policy:int, token: str):
	url = 'https://celery.mooclet.com/engine/api/v1/value'
	data = json.dumps({'variable': variable, 'value': value, 'learner': user_id, 'mooclet': mooclet, 'version': version, 'policy': policy})
	data = data.encode('utf-8')
	header = {'Authorization': 'Token ' + token, 'Content-Type': 'application/json', 'Content-Length': len(data)}
	return open_url_to_dict(url, data, 'POST', header)

if __name__ == "__main__":
	data_path = './data/q5easy1ans_mooclet.csv'
	past_mooclet = {'mooclet': 145, 'versions': [5197, 5198, 5199], 'outcome_variable_name': 'q5easy1ans'}
	sim_mooclet = {'mooclet': 153, 'versions': [5215, 5216, 5217], 'outcome_variable_name': 'finalstr7_replica_rwd'}
	version_mapping = {5197: 5215, 5198: 5216, 5199: 5217}
	token = '1f036c60881ea71e4adec412ce931cfac9926373'
	sim_count = 1000
	num_processes = 100 # must devide sim_count
	factor = 'finalstr7'
	out_folder = './output/{}_online/'.format(factor)
	csv_path = '{}{}_models_online.csv'.format(out_folder, factor)
	stops = [10, 100, 368] # must not be larger than input TS data
	levels = ['nothing', 'flowchart', 'additional problem'] # description of versions
	wait_time = 180 # wait for this seconds when the server crashes
	timeout = 15 # timeout in this seconds when checking server load

	###################### You don't have to change below ######################

	# You may want to change this if reward is not binary or you want to try different priors
	context = {'policy_parameters':
				{'outcome_variable_name': past_mooclet['outcome_variable_name'], 'max_rating': 1, 'prior':{'success': 1, 'failure': 1}}, 
			'mooclet': past_mooclet['mooclet'], 'versions': past_mooclet['versions']}

	# Read csv and extract TS data
	data = pd.read_csv(data_path)
	data = data.dropna(subset=['version_id_{}'.format(past_mooclet['mooclet'])]) # This won't change alpha and beta
	data = data[data['policy_id_{}'.format(past_mooclet['mooclet'])] == 3].sort_values(
		'timestamp_{}'.format(past_mooclet['mooclet']), ignore_index=True)
	data = data.where(data.notnull(), None)

	# Create columns for output
	columns = []
	for version in sim_mooclet['versions']:
		columns.append('probability_{}'.format(version))
	for version in past_mooclet['versions']:
		columns.append('successes_{}'.format(version))
		columns.append('failures_{}'.format(version))
	df = pd.DataFrame(columns=columns)

	# Run simulations
	for i in range(data.shape[0]):
		# Get assignment probability of each arm empirically
		jobs = []
		sim_result = {'probability_{}'.format(version): 0 for version in sim_mooclet['versions']}
		q = multiprocessing.Queue()
		for _ in range(num_processes):
			p = multiprocessing.Process(target=get_arm_parallel, args=(
				sim_mooclet['mooclet'], 'learner', token, sim_mooclet['versions'], sim_count // num_processes, wait_time, q))
			jobs.append(p)
			p.start()
		for proc in jobs:
			proc.join()
		count = 0
		while not q.empty():
			returned_dict = q.get()
			count += 1
			for version in sim_result.keys():
				sim_result[version] += returned_dict[version]
		if count != num_processes:
			print('There should be {} processes but only {} of them returned a desired output'.format(num_processes, count))
		for key in sim_result.keys():
			sim_result[key] /= sim_count * count / num_processes
		
		# Increment the number of data points one by one and save alpha and beta to a DataFrame
		simulation_data = data[:i]
		sim_result.update(thompson_sampling([], context, simulation_data))
		df = df.append(sim_result.copy(), ignore_index=True)

		# Send reward to mooclets
		cur_user = data.iloc[i]
		version = cur_user['version_id_{}'.format(past_mooclet['mooclet'])]
		if version != None:
			version = version_mapping[version]
		response = None
		while response == None:
			try:
				response = send_reward(sim_mooclet['outcome_variable_name'], float(cur_user[past_mooclet['outcome_variable_name']]), 
					str(cur_user['StudentRandomID']), sim_mooclet['mooclet'], version, 3, token)
			except (urllib.error.HTTPError, urllib.error.URLError) as e:
				print(str(e))
				sleep(wait_time)
		print('{}th reward was sent!'.format(i+1))

		# Check server load by limiting timeout
		response = None
		while response == None:
			try:
				response = get_arm(sim_mooclet['mooclet'], 'learner', token, timeout)
			except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout) as e:
				print("Probabily timed out. Let's wait for {} seconds and try again.".format(wait_time))
				sleep(wait_time)


	# Save the results
	df.to_csv(csv_path)

	# Plot
	ap_lists = []
	for version in sim_mooclet['versions']:
		ap_lists.append(df['probability_{}'.format(version)])
	for stop in stops:
		plot_assignment_prob(ap_lists, levels, stop, factor, out_folder)
		plot_success_failure(df, past_mooclet['versions'], levels, stop, factor, out_folder)
		for i in range(len(past_mooclet['versions'])):
			plot_beta_dist(df['successes_{}'.format(past_mooclet['versions'][i])], df['failures_{}'.format(past_mooclet['versions'][i])], 
				stop, levels[i], out_folder)
