import string

from numpy.random import choice, beta
#from django.core.urlresolvers import reverse
#from django.apps import apps
# from django.contrib.contenttypes.models import ContentType
#from django.db.models import Avg
import json
from collections import Counter
# from .utils.utils import sample_no_replacement
#from django.db.models.query_utils import Q
import datetime
import numpy as np
from scipy.stats import invgamma
#from django.forms.models import model_to_dict
import re
# arguments to policies:

# variables: list of variable objects, can be used to retrieve related data
# context: dict passed from view, contains current user, course, quiz, question context

def thompson_sampling_placeholder(variables,context):
	return choice(context['mooclet'].version_set.all())

def thompson_sampling(variables,context,simulation_data):
	'''
	versions = context['mooclet'].version_set.all()
	#import models individually to avoid circular dependency
	Variable = apps.get_model('engine', 'Variable')
	Value = apps.get_model('engine', 'Value')
	Version = apps.get_model('engine', 'Version')
	# version_content_type = ContentType.objects.get_for_model(Version)
	#priors we set by hand - will use instructor rating and confidence in future
	# TODO : all explanations are having the same prior.
	'''
	versions = context['versions']

	# context is the following json :
	#   {
	#   'policy_parameters':
	#       {
	#       'outcome_variable_name':<name of the outcome variable',
	#       'max_rating': <maximum value of the outcome variable>,
	#       'prior':
	#           {'success':<prior success value>},
	#           {'failure':<prior failure value>},
	#       }
	#   }
	policy_parameters = context["policy_parameters"]#.parameters

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	#max value of version rating, from qualtrics
	max_rating = policy_parameters['max_rating']

	version_to_show = None
	max_beta = 0
	alpha_beta = {}

	for version in versions:
		'''
		if "used_choose_group" in context and context["used_choose_group"] == True:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet'], 'policy': 'thompson_sampling'})
		else:
			student_ratings = Variable.objects.get(name=outcome_variable_name).get_data(context={'version': version, 'mooclet': context['mooclet']})
		'''
		student_ratings = simulation_data[simulation_data['version_id_{}'.format(context['mooclet'])] == int(version)][outcome_variable_name]

		'''
		if student_ratings:
			student_ratings = student_ratings.all()
			rating_count = student_ratings.count()
			rating_average = student_ratings.aggregate(Avg('value'))
			rating_average = rating_average['value__avg']
			if rating_average is None:
				rating_average = 0

		else: 
			rating_average = 0
			rating_count = 0
		'''
		if student_ratings.empty:
			rating_average = 0
			rating_count = 0
		else:
			rating_count = student_ratings.size
			rating_average = student_ratings.mean()

		#TODO - log to db later?
		successes = (rating_average * rating_count) + prior_success
		failures = (max_rating * rating_count) - (rating_average * rating_count) + prior_failure
		'''
		print("successes: " + str(successes))
		print("failures: " + str(failures))
		version_beta = beta(successes, failures)

		if version_beta > max_beta:
			max_beta = version_beta
			version_to_show = version
		'''
		alpha_beta['successes_{}'.format(version)] = successes
		alpha_beta['failures_{}'.format(version)] = failures

	return alpha_beta
