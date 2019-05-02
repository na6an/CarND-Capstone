from styx_msgs.msg import TrafficLight

import os
import sys
import time
from PIL import Image

import numpy as np
import tensorflow as tf

import rospy

DETECTION_THRESHOLD = 0.8

class TLClassifier(object):
    def __init__(self, model_path):
        # Import tensorflow graph
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
		
            # get all necessary tensors
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')            

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

            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            
            # run classifier
            (scores, classes) = self.sess.run(
                [self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded})
            
            # find the top score for a given image frame
            idx = np.argmax(np.squeeze(scores))
            top_score = np.squeeze(scores)[idx]
            
            elapsed_time = time.time() - tic
            rospy.loginfo("Time spent on classification=%.2f sec" % (elapsed_time))

            # figure out traffic light class based on the top score
            if top_score > DETECTION_THRESHOLD:
                tl_state = int(np.squeeze(classes)[idx])
                
                if tl_state == 1:                    
                    return TrafficLight.RED
                elif tl_state == 2:                    
                    return TrafficLight.YELLOW
                else:
                    return TrafficLight.GREEN
            else:                
                return TrafficLight.UNKNOWN
