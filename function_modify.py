
def analyze(img_path, actions = ['emotion', 'age', 'gender', 'race'] , models = {}, enforce_detection = True, detector_backend = 'opencv', prog_bar = True,align = True):

	"""
	This function analyzes facial attributes including age, gender, emotion and race

	Parameters:
		img_path: exact image path, numpy array or base64 encoded image could be passed. If you are going to analyze lots of images, then set this to list. e.g. img_path = ['img1.jpg', 'img2.jpg']

		actions (list): The default is ['age', 'gender', 'emotion', 'race']. You can drop some of those attributes.

		models: facial attribute analysis models are built in every call of analyze function. You can pass pre-built models to speed the function up.

			models = {}
			models['age'] = DeepFace.build_model('Age')
			models['gender'] = DeepFace.build_model('Gender')
			models['emotion'] = DeepFace.build_model('Emotion')
			models['race'] = DeepFace.build_model('race')

		enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.

		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib.

		prog_bar (boolean): enable/disable a progress bar
	Returns:
		The function returns a dictionary. If img_path is a list, then it will return list of dictionary.

		{
			"region": {'x': 230, 'y': 120, 'w': 36, 'h': 45},
			"age": 28.66,
			"gender": "woman",
			"dominant_emotion": "neutral",
			"emotion": {
				'sad': 37.65260875225067,
				'angry': 0.15512987738475204,
				'surprise': 0.0022171278033056296,
				'fear': 1.2489334680140018,
				'happy': 4.609785228967667,
				'disgust': 9.698561953541684e-07,
				'neutral': 56.33133053779602
			}
			"dominant_race": "white",
			"race": {
				'indian': 0.5480832420289516,
				'asian': 0.7830780930817127,
				'latino hispanic': 2.0677512511610985,
				'black': 0.06337375962175429,
				'middle eastern': 3.088453598320484,
				'white': 93.44925880432129
			}
		}

	"""

	img_paths, bulkProcess = functions.initialize_input(img_path)

	#---------------------------------

	built_models = list(models.keys())

	#---------------------------------

	#pre-trained models passed but it doesn't exist in actions
	if len(built_models) > 0:
		if 'emotion' in built_models and 'emotion' not in actions:
			actions.append('emotion')

		if 'age' in built_models and 'age' not in actions:
			actions.append('age')

		if 'gender' in built_models and 'gender' not in actions:
			actions.append('gender')

		if 'race' in built_models and 'race' not in actions:
			actions.append('race')

	#---------------------------------

	if 'emotion' in actions and 'emotion' not in built_models:
		models['emotion'] = build_model('Emotion')

	if 'age' in actions and 'age' not in built_models:
		models['age'] = build_model('Age')

	if 'gender' in actions and 'gender' not in built_models:
		models['gender'] = build_model('Gender')

	if 'race' in actions and 'race' not in built_models:
		models['race'] = build_model('Race')

	#---------------------------------

	resp_objects = []

	disable_option = (False if len(img_paths) > 1 else True) or not prog_bar

	global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing', disable = disable_option)

	for j in global_pbar:
		img_path = img_paths[j]

		resp_obj = {}

		disable_option = (False if len(actions) > 1 else True) or not prog_bar

		pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable = disable_option)

		img_224 = None # Set to prevent re-detection

		region = [] # x, y, w, h of the detected face region
		region_labels = ['x', 'y', 'w', 'h']

		is_region_set = False

		#facial attribute analysis
		for index in pbar:
			action = actions[index]
			pbar.set_description("Action: %s" % (action))

			if action == 'emotion':
				emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
				img, region = functions.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True,align = align)

				emotion_predictions = models['emotion'].predict(img)[0,:]

				sum_of_predictions = emotion_predictions.sum()

				resp_obj["emotion"] = {}

				for i in range(0, len(emotion_labels)):
					emotion_label = emotion_labels[i]
					emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
					resp_obj["emotion"][emotion_label] = emotion_prediction

				resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

			elif action == 'age':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True,align = align)

				age_predictions = models['age'].predict(img_224)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)

				resp_obj["age"] = int(apparent_age) #int cast is for the exception - object of type 'float32' is not JSON serializable

			elif action == 'gender':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True,align = align)

				gender_prediction = models['gender'].predict(img_224)[0,:]

				if np.argmax(gender_prediction) == 0:
					gender = "Woman"
				elif np.argmax(gender_prediction) == 1:
					gender = "Man"

				resp_obj["gender"] = gender

			elif action == 'race':
				if img_224 is None:
					img_224, region = functions.preprocess_face(img = img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True,align = align) #just emotion model expects grayscale images
				race_predictions = models['race'].predict(img_224)[0,:]
				race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

				sum_of_predictions = race_predictions.sum()

				resp_obj["race"] = {}
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions
					resp_obj["race"][race_label] = race_prediction

				resp_obj["dominant_race"] = race_labels[np.argmax(race_predictions)]

			#-----------------------------

			if is_region_set != True:
				resp_obj["region"] = {}
				is_region_set = True
				for i, parameter in enumerate(region_labels):
					resp_obj["region"][parameter] = int(region[i]) #int cast is for the exception - object of type 'float32' is not JSON serializable

		#---------------------------------

		if bulkProcess == True:
			resp_objects.append(resp_obj)
		else:
			return resp_obj

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["instance_%d" % (i+1)] = resp_item

		return resp_obj