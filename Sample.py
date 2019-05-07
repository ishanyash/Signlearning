##########################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
##########################################################################

import Leap
import sys
import thread
import time
import numpy as np
import math
import json
import pprint
import random
from sets import Set
captureData = False
# captureData = True


# from PIL import Image
# import ctypes
import matplotlib.pyplot as plt
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']

    frameList = []

    def sigmoid(self, x):
        s = 1.0 / (1.0 + np.exp(-1.0 * x))
        return s

    def predict(self, features, weights):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        z = np.dot(features, weights)
        return self.sigmoid(z)

    def dumpList(self):
        with open("E_data_pos.json", "w") as f:
            json.dump(self.frameList, f)

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def angleBetween(self, v1, v2):
        V1 = np.array([v1[0], v1[1], v1[2]])
        V2 = np.array([v2[0], v2[1], v2[2]])
        x = np.arccos(np.dot(V1, V2)) * Leap.RAD_TO_DEG
        if(math.isnan(x)):
            return 0
        else:
            return x

    def angleBetweenAllFingers(self, fingers):
        table = np.zeros((5, 5))
        table.astype(float)
        for i in range(0, len(fingers)):
            for j in range(0, len(fingers)):
                table[i][j] = self.angleBetween(
                    fingers[i].direction, fingers[j].direction)
        return table

    def boneAnglesOneFinger(self, finger):
        row = np.zeros((1, 3))
        row.astype(float)
        for i in range(0, len(row[0])):
            x = self.angleBetween(finger.bone(i).direction,
                                  finger.bone(i + 1).direction)
            if(math.isnan(x)):
                x = 0
            else:
                row[0][i] = x

        return row

    def boneAnglesAllFingers(self, fingers):
        table = np.zeros((5, 3))
        table.astype(float)
        for i in range(0, len(fingers)):
            row = self.boneAnglesOneFinger(fingers[i])
            for j in range(0, len(table[0])):
                table[i][j] = row[0][j]
        return table

    def accessBone_dot_table(self, finger, bone, table):
        if(bone == 0):
            return -1
        else:
            return table[finger][bone - 1]

    def isFingerStraight(self, finger, table):
        distal_to_inter_threshold = 10
        inter_to_proximal_threshold = 10
        proximal_to_meta_threshold = 40

        proximal = 1
        intermediate = 2
        distal = 3

        if(self.accessBone_dot_table(finger, distal, table) < distal_to_inter_threshold and self.accessBone_dot_table(finger, intermediate, table) < inter_to_proximal_threshold and self.accessBone_dot_table(finger, proximal, table) < proximal_to_meta_threshold):
            return True
        else:
            return False

    def allFingerStraight(self, table):
        straightTable = np.zeros((len(table), 1))
        for i in range(0, len(table)):
            straightTable[i] = self.isFingerStraight(i, table)
        return straightTable

    def doubleCheckThumbStraight(self, finger, table):
        proximal = 1
        intermediate = 2

        distal_to_inter_threshold = 15

        dot = self.angleBetween(finger.bone(
            intermediate).direction, finger.bone(proximal).direction)
        if(dot < distal_to_inter_threshold):
            table[0] = True
        else:
            table[0] = False

    def distanceBetween(self, v1, v2):
        V1 = np.array([v1[0], v1[1], v1[2]])
        V2 = np.array([v2[0], v2[1], v2[2]])
        d = V1 - V2

        return (d[0]**2 + d[1]**2 + d[2]**2) ** 0.5

    def distanceBetweenAllFingerTips(self, fingers):
        table = np.zeros((5, 5))
        table.astype(float)
        distal = 3
        for i in range(0, len(fingers)):
            for j in range(0, len(fingers)):
                table[i][j] = self.distanceBetween(fingers[i].bone(
                    distal).center, fingers[j].bone(distal).center)
        return table

    def score(self, value, avg, std):
        # value = 108.12540506134567
        # avg = 142.88036222182996
        # std = 32.08512792599652
        print("value : ", value)
        print("average: ", avg)
        print("std dev: ", std)
        if(value == avg):
            return 1
        if((value >= avg + (std * 2)) or (value <= avg - (std * 2))):
            return 0
        val = abs(value - avg)
        scrVal = (-1.0 / (2.0 * val)) * val + 1
        # print("src:", scrVal)
        if(scrVal < 0):
            return 0
        else:
            return scrVal

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d, tools: %d, gestures: %d" % (
        #       frame.id, frame.timestamp, len(frame.hands), len(frame.fingers), len(frame.tools), len(frame.gestures()))
        for hand in frame.hands:
            handType = "Left hand" if hand.is_left else "Right hand"
#            print "  %s, id %d, position: %s" % (
#                handType, hand.id, hand.palm_position)

            # Get the hand's normal vector and direction
            normal = hand.palm_normal
            direction = hand.direction

            # Calculate the hand's pitch, roll, and yaw angles
#            print "  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
#                direction.pitch * Leap.RAD_TO_DEG,
#                normal.roll * Leap.RAD_TO_DEG,
#                direction.yaw * Leap.RAD_TO_DEG)

            # Get arm bone
#            arm = hand.arm
#            print "  Arm direction: %s, wrist position: %s, elbow position: %s" % (
#                arm.direction,
#                arm.wrist_position,
#                arm.elbow_position)

            # FINGER INDEXING
            thumb = 0
            index = 1
            middle = 2
            ring = 3
            pinky = 4
            # BONE INDEXING
            metacarpal = 0
            proximal = 1
            intermediate = 2
            distal = 3

            # DOT PRODUCT BETWEEN THE DIRECTION EACH FINGERTIP IS POINTED
            # N x N matrix  where N is the number of fingers.
            # use finger types to access the dot product you want
            finger_dot_table = self.angleBetweenAllFingers(hand.fingers)

            # DISTANCE BETWEEN THE CENTER OF THE DISTAL BONE IN EACH FINGER
            # N x N matrix  where N is the number of fingers.
            # use finger type to access distance you want
            fingertip_distance_table = self.distanceBetweenAllFingerTips(
                hand.fingers)

            # DOT PRODUCT OF EACH BONE IN THE FINGER
            # N x M matrix  where N is the number of fingers & M is number of dot products
            # metacarpal->proximal->intermediate->distal
            # 3 total dot products (one for each arrow above)
            bone_dot_table = self.boneAnglesAllFingers(hand.fingers)
            # print(bone_dot_table)

            # WHETHER OR NOT THE GIVEN FINGER IS STRAIGHT
            # 1 = True | 0 = False
            # First check when comparing input to knowledge base!
            # Maybe take a quantity samples of input to allow sensor to dectect
            # finger orientation.
            finger_straight_table = self.allFingerStraight(bone_dot_table)
            # HANDLE THUMB WHICH HAS DIFFERENT STRUCTURE THAN 4 FINGERS
            self.doubleCheckThumbStraight(
                hand.fingers[thumb], finger_straight_table)
            # print("After thumb check:")
            # print(finger_straight_table)

            if(captureData):
                data_out = {}
                data_out['finger_dot_table'] = (
                    np.ndarray.flatten(finger_dot_table)).tolist()
                data_out['fingertip_distance_table'] = (
                    np.ndarray.flatten(fingertip_distance_table)).tolist()
                data_out['bone_dot_table'] = (
                    np.ndarray.flatten(bone_dot_table)).tolist()
                data_out['finger_straight_table'] = (
                    np.ndarray.flatten(finger_straight_table)).tolist()

                if(frame.id % 10 == 0):
                    print("capture")
                    self.frameList.append(data_out)

                    #pp = pprint.PrettyPrinter(indent='4')
                    # pp.pprint(listener.frameList)
                    # with open("L_data.json", "a") as f:
                    #     json.dump(data_out, f)

            if(not captureData):
                # with open("L_data.json") as f:
                #     data = json.load(f)
                with open("E_dot_weights.json") as f1:
                    weights_data = json.load(f1)



                # score = 0
                temp = np.ndarray.flatten(finger_dot_table)

                # print("weight:", weights_data[0])
                pred = self.predict(temp, weights_data)
                print("prediction: ", pred[0])
                # for i in range(0, len(temp)):
                #     tempScore = self.score(temp[i], data['finger_dot_table_avg'][
                #                            i], data['finger_dot_table_std'][i])
                #     score = score + tempScore
                #     print("tempScore:", tempScore)
                #     print("totalscore : ", score)

                # score = score / len(temp)

                # print(score)

                # if(score > 0.6):
                #     print("recognized: ")
                #     print(data['letter'])


# TOOLS AND BUILT IN GESTURE DETECTION #

        # # Get tools
        # for tool in frame.tools:

        #     print "  Tool id: %d, position: %s, direction: %s" % (
        #         tool.id, tool.tip_position, tool.direction)

        # # Get gestures
        # for gesture in frame.gestures():
        #     if gesture.type == Leap.Gesture.TYPE_CIRCLE:
        #         circle = CircleGesture(gesture)

        #         # Determine clock direction using the angle between the pointable and the circle normal
        #         if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
        #             clockwiseness = "clockwise"
        #         else:
        #             clockwiseness = "counterclockwise"

        #         # Calculate the angle swept since the last frame
        #         swept_angle = 0
        #         if circle.state != Leap.Gesture.STATE_START:
        #             previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
        #             swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI

        #         print "  Circle id: %d, %s, progress: %f, radius: %f, angle: %f degrees, %s" % (
        #                 gesture.id, self.state_names[gesture.state],
        # circle.progress, circle.radius, swept_angle * Leap.RAD_TO_DEG,
        # clockwiseness)

        #     if gesture.type == Leap.Gesture.TYPE_SWIPE:
        #         swipe = SwipeGesture(gesture)
        #         print "  Swipe id: %d, state: %s, position: %s, direction: %s, speed: %f" % (
        #                 gesture.id, self.state_names[gesture.state],
        #                 swipe.position, swipe.direction, swipe.speed)

        #     if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
        #         keytap = KeyTapGesture(gesture)
        #         print "  Key Tap id: %d, %s, position: %s, direction: %s" % (
        #                 gesture.id, self.state_names[gesture.state],
        #                 keytap.position, keytap.direction )

        #     if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
        #         screentap = ScreenTapGesture(gesture)
        #         print "  Screen Tap id: %d, %s, position: %s, direction: %s" % (
        #                 gesture.id, self.state_names[gesture.state],
        #                 screentap.position, screentap.direction )

        # if not (frame.hands.is_empty and frame.gestures().is_empty):
        #     print ""
    def generateRandIndices(self, size, indexRange):
        indices = []
        while(len(indices) < size and indexRange > size):
            r = random.randint(0, indexRange)
            if r not in indices:
                indices.append(r)

        return np.sort(indices, kind='mergesort')

    def elementStatistics(self, elements):
        print("length of elements:", len(elements))
        print(elements[0])
        if(len(elements) == 0):
            return (0, 0)

        std = np.std(elements)
        avg = np.average(elements)

        return(avg, std)

    def getAllElements(self, table_data, elementIndex):
        elements = []
        for i in range(0, len(table_data)):
            elements.append(table_data[i][elementIndex])
        return elements

    def selectElements(self, indices, table_data, elementIndex):
        elements = []
        for index in indices:
            elements.append(table_data[index][elementIndex])

        return elements

    def generateTableHolder(self, fl, tableKey):
        tableHolder = []
        for i in range(0, len(fl)):
            tableHolder.append(fl[i][tableKey])
        return tableHolder

    # RANSAC Algorithm for removing outlier values
    def removeOutliers(self, fl, test, num):
        index = self.generateRandIndices(num, len(test) - 1)
        outliers = Set()
        # repeat N times to ensure high probability of removing outliers
        # N = (len(test) * 10) / num
        # for k in range(0,N):
        #     print("iteration = " , k)

        for i in range(0, len(test[0])):
            elements = self.selectElements(index, test, i)
            stats = self.elementStatistics(elements)
            print("Average: ", stats[0])
            print("STd dev: ", stats[1])
            for j in range(0, len(index)):
                x = test[index[j]][i]
                if(x > (stats[0] + (stats[1] * 2)) or x < (stats[0] - (stats[1] * 2))):
                    # remove that frame test[index[j]]
                    print(x)
                    outliers.add(index[j])
        # AFTER ONE ITERATION OF SAMPLING, REMOVE OUTLIER FRAMES
            # rm = list(outliers)
            # rm.sort(reverse = True)
            # print("indices ", index)
            # print("outliers" , rm)
            # for i in range(0, len(rm)):
            #     del fl[rm[i]]
        outliers = list(outliers)
        outliers.sort(reverse=True)
        print("indices ", index)
        print("outliers", outliers)
        print(len(fl))
        for i in range(0, len(outliers)):
            del fl[outliers[i]]

    def avgData(self, fl, test):
        avg = []
        std = []
        for i in range(0, len(test[0])):
            print("I val : ", i)
            print("length of test: ", len(test))
            print(" length of test table = ", len(test[0]))
            elements = self.getAllElements(test, i)
            stats = self.elementStatistics(elements)
            for j in range(0, 4):
                print(elements[j])
            avg.append(stats[0])
            std.append(stats[1])
        return avg, std

    def state_string(self, state):
        if state == Leap.Gesture.STATE_START:
            return "STATE_START"

        if state == Leap.Gesture.STATE_UPDATE:
            return "STATE_UPDATE"

        if state == Leap.Gesture.STATE_STOP:
            return "STATE_STOP"

        if state == Leap.Gesture.STATE_INVALID:
            return "STATE_INVALID"


def main():

    # Create a sample listener and controller
    listener = SampleListener()
    #listener2 = SampleListener()
    controller = Leap.Controller()
    #controller2 = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    # controller2.add_listener(listener2)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
        if(captureData):
            listener.dumpList()



    # if(captureData):

    #     # ACTUAL
    #     fl = listener.frameList
    #     print("Start123")
    #     print("FL size: ", len(fl))

        # num = 10
        # N = (len(fl)) / num
        # for i in range(0, N):
        #     # fst = listener.generateTableHolder(fl, 'finger_straight_table')
        #     # listener.removeOutliers(fl, fst, num)

        #     fdt = listener.generateTableHolder(fl, 'finger_dot_table')
        #     listener.removeOutliers(fl, fdt, num)

        #     bdt = listener.generateTableHolder(fl, 'bone_dot_table')
        #     listener.removeOutliers(fl,bdt, num)

        #     ftdt = listener.generateTableHolder(fl, 'fingertip_distance_table')
        #     listener.removeOutliers(fl, ftdt, num)

        # print("FL size after: " , len(fl))

        # fst = listener.generateTableHolder(fl, 'finger_straight_table')
        # fdt = listener.generateTableHolder(fl, 'finger_dot_table')
        # bdt = listener.generateTableHolder(fl, 'bone_dot_table')
        # ftdt = listener.generateTableHolder(fl, 'fingertip_distance_table')
        # fst_avg,fst_std = listener.avgData(fl,fst)
        # fdt_avg,fdt_std = listener.avgData(fl,fdt)
        # bdt_avg,bdt_std = listener.avgData(fl,bdt)
        # ftdt_avg,ftdt_std = listener.avgData(fl,ftdt)

        # data_out = {}
        # data_out['finger_straight_table_avg'] = fst_avg
        # data_out['finger_straight_table_std'] = fst_std
        # data_out['finger_dot_table_avg'] = fdt_avg
        # data_out['finger_dot_table_std'] = fdt_std
        # data_out['bone_dot_table_avg'] = bdt_avg
        # data_out['bone_dot_table_std'] = bdt_std
        # data_out['fingertip_distance_table_avg'] = ftdt_avg
        # data_out['fingertip_distance_table_std'] = ftdt_std
        # data_out['letter'] = "l"

        # #pp = pprint.PrettyPrinter(indent='4')
        # #pp.pprint(listener.frameList)
        # with open("L_data.json", "w") as f:
        #                 json.dump(data_out, f)


if __name__ == "__main__":
    main()
