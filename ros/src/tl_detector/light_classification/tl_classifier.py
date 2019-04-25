from styx_msgs.msg import TrafficLight

import os
import sys
import time
from PIL import Image

import numpy as np
import tensorflow as tf

import rospy

DETECTION_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.is_site = False

        # specify path to /models directory with respect to the absolute path of tl_classifier.py
        model_dir=os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # specify the model name based on the is_site flag state
        if self.is_site:    
            file_name = 'frnn_real.pb'
        else:
            file_name = 'ssd_sim.pb'

        # full path to the model file
        frozen_graph_file = os.path.join(model_dir, file_name)

        # Import tensorflow graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
		
            # get all necessary tensors
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.detection_graph)
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Bounding Box Detection.
        tic = time.time()
        with self.detection_graph.as_default():
            # BGR to RGB conversion
            image = image[:, :, ::-1]

            img = Image.fromarray(image.astype('uint8'), 'RGB')            
            img.thumbnail((400, 300), Image.ANTIALIAS)
            
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            # run classifier
            (scores, classes) = self.sess.run(
                [self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded})
            
            # find the top score for a given image frame
            top_score = np.amax(np.squeeze(scores))
            
            elapsed_time = time.time() - tic
            rospy.logdebug("Time spent on classification=%.2f sec" % (elapsed_time))

            # figure out traffic light class based on the top score
            if top_score > DETECTION_THRESHOLD:
                tl_state = int(np.squeeze(classes)[0])
                
                if tl_state == 1:                    
                    return TrafficLight.RED
                elif tl_state == 2:
                    rospy.logdebug("Traffic state: YELLOW, score=%.2f" % (top_score*100))
                    return TrafficLight.YELLOW
                else:
                    rospy.logdebug("Traffic state: GREEN, score=%.2f" % (top_score*100))
                    return TrafficLight.GREEN
            else:
                rospy.logdebug("Traffic state: OFF")
                return TrafficLight.UNKNOWN
