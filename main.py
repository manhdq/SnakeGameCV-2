import cvzone
import cv2
import math
import os
import time
import numpy as np
import pandas as pd
import random
import argparse
from cvzone.HandTrackingModule import HandDetector


class SnakeGameClass:
    def __init__(self, opt):
        self.opt = opt
        self.imgSFood = cv2.imread(opt.pathSFood, cv2.IMREAD_UNCHANGED)
        self.imgSFood = cv2.resize(self.imgSFood, tuple(opt.SFoodSize))
        self.imgLFood = cv2.imread(opt.pathLFood, cv2.IMREAD_UNCHANGED)
        self.imgLFood = cv2.resize(self.imgLFood, tuple(opt.LFoodSize))
        self.reset()

        self.gameOver = False
        self.menuScreen = True
        self.inGame = False
        self.level = 0
        # self.levelDict = {0: 'Easy', 1: 'Medium', 2: 'Hard'}
        self.inClick = False

        self.dataIndividualDict = {
            'HoTen': [],
            'Khoa': [],
            'Nganh': [],
            'MSSV': [],
            'Score': [],
            'Option': [],
        }

        self.dataDict = pd.read_csv(self.opt.main_csv_file).to_dict('list')

    def reset(self):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 50  # total allowed length
        self.previousHead = 0, 0  # previous head point
        self.distanceThresh = 300  # maximum distance between each point
        self.score = 0  # Initial score
        self.startTime = time.time()
        self.endTime = 0.0
        self.countdown = 3
        self.dying = 0
        self.waitPlaying = 10
        self.waitCount = 3
        self.playing = False

        self.randomSFoodLocation()
        self.randomLFoodLocation()

        self.lengthPerSFood = 150
        self.scorePerSFood = 100
        self.lengthPerLFood = 0
        self.scorePerLFood = 1000
        self.LFoodExist = False
        self.LFoodCount = 5
        self.LFoodWait = 30

    def getScreen(self):
        if self.opt.pathScreen is not None:
            imgInitial = cv2.imread(self.opt.pathScreen)[..., ::-1]
            imgInitial = cv2.resize(imgInitial, (self.opt.imgWidth, self.opt.imgHeight))
        else:
            imgInitial = np.zeros((self.opt.imgHeight, self.opt.imgWidth, 3))

        ## Game Name
        font = cv2.FONT_HERSHEY_COMPLEX
        text = self.opt.name
        textsize = cv2.getTextSize(text, font, 3, 9)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 200), font, 3, (0, 0, 0), 9)
        textsize = cv2.getTextSize(text, font, 3, 5)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 200), font, 3, (0, 200, 0), 5)

        img = imgInitial.copy()

        ## Game Option  (0-640)
        # Start game
        img = cv2.rectangle(img, (100, 300), (540, 400), (0, 200, 0), -1)
        font = cv2.FONT_HERSHEY_COMPLEX
        text = "Start Game"
        textsize = cv2.getTextSize(text, font, 1, 3)[0]
        img = cv2.putText(img, text, (int((img.shape[1] - 640 - textsize[0]) / 2), 360), font, 1, (0, 0, 0), 3)
        # Level
        img = cv2.rectangle(img, (100, 420), (540, 520), (0, 200, 0), -1)
        font = cv2.FONT_HERSHEY_COMPLEX
        text = f"Test"
        textsize = cv2.getTextSize(text, font, 1, 3)[0]
        img = cv2.putText(img, text, (int((img.shape[1] - 640 - textsize[0]) / 2), 480), font, 1, (0, 0, 0), 3)


        # Combine
        img = cv2.addWeighted(img, 0.7, imgInitial, 0.3, 1.0)

        return img

    def updateScreen(self, img, points):
        c1x, c1y = points[0]
        c2x, c2y = points[1]
        distance = math.hypot(c1x-c2x, c1y-c2y)
        # print(distance)

        if distance < 45 and self.inClick == False:  # in-click for
            self.inClick = True
        if distance > 60 and self.inClick == True:
            self.inClick = False
            cx, cy = (c1x + c2x) // 2, (c1y + c2y) // 2

            ## Click start game
            if 100 < cx < 540 and 300 < cy < 400:
                self.menuScreen = False
                self.inGame = True
                self.startGame = True
                self.testGame = False
            ## Click test game
            if 100 < cx < 540 and 420 < cy < 520:
                self.menuScreen = False
                self.inGame = True
                self.startGame = False
                self.testGame = True

            # elif 100 < cx < 540 and 420 < cy < 520:
            #     self.level = (self.level + 1) % len(self.levelDict)

    def getGameOver(self):
        if self.opt.pathGameOver is not None:
            imgInitial = cv2.imread(self.opt.pathGameOver)[..., ::-1]
            imgInitial = cv2.resize(imgInitial, (self.opt.imgWidth, self.opt.imgHeight))
        else:
            imgInitial = np.ones((self.opt.imgHeight, self.opt.imgWidth, 3)) * 255

        ## Game Name
        font = cv2.FONT_HERSHEY_COMPLEX
        text = "GAME OVER"
        textsize = cv2.getTextSize(text, font, 3, 9)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 180), font, 3, (0, 0, 0), 9)
        textsize = cv2.getTextSize(text, font, 3, 5)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 180), font, 3, (0, 200, 0), 5)

        font = cv2.FONT_HERSHEY_COMPLEX
        text = self.opt.playerName
        textsize = cv2.getTextSize(text, font, 3, 9)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 280), font, 3, (0, 0, 0), 9)
        textsize = cv2.getTextSize(text, font, 3, 5)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 280), font, 3, (0, 200, 0), 5)

        font = cv2.FONT_HERSHEY_COMPLEX
        text = f"Score: {self.score}"
        textsize = cv2.getTextSize(text, font, 3, 9)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 380), font, 3, (0, 0, 0), 9)
        textsize = cv2.getTextSize(text, font, 3, 5)[0]
        imgInitial = cv2.putText(imgInitial, text, (int((imgInitial.shape[1] - textsize[0]) / 2), 380), font, 3, (0, 200, 0), 5)

        img = imgInitial.copy()

        ## Game Option  (0-640)
        # Start game
        img = cv2.rectangle(img, (300, 480), (980, 620), (0, 200, 0), -1)
        font = cv2.FONT_HERSHEY_COMPLEX
        text = "Return to Menu"
        textsize = cv2.getTextSize(text, font, 2, 3)[0]
        img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 570), font, 2, (0, 0, 0), 3)


        # Combine
        img = cv2.addWeighted(img, 0.7, imgInitial, 0.3, 1.0)

        return img

    def updateGameOver(self, img, points):
        c1x, c1y = points[0]
        c2x, c2y = points[1]
        distance = math.hypot(c1x-c2x, c1y-c2y)
        # print(distance)

        if distance < 45 and self.inClick == False:  # in-click
            self.inClick = True
        if distance > 60 and self.inClick == True:
            self.inClick = False
            cx, cy = (c1x + c2x) // 2, (c1y + c2y) // 2

            ## Click start game
            if 300 < cx < 980 and 480 < cy < 620:
                self.menuScreen = True
                self.gameOver = False
                self.reset()

    def randomSFoodLocation(self):
        self.foodSPoint = random.randint(300+self.opt.SFoodSize[0],980-self.opt.SFoodSize[0]), random.randint(200+self.opt.SFoodSize[1],520-self.opt.SFoodSize[1])

    def randomLFoodLocation(self):
        self.foodLPoint = random.randint(300+self.opt.LFoodSize[0],980-self.opt.LFoodSize[0]), random.randint(200+self.opt.LFoodSize[1],520-self.opt.LFoodSize[1])

    def getInGame(self, currentHead):
        img = cv2.imread(self.opt.pathInGameBackground, )
        img = cv2.resize(img, (self.opt.imgWidth, self.opt.imgHeight))

        logo = cv2.imread(self.opt.pathLogo, )
        logo = cv2.resize(logo, (160, 160))

        if not self.playing:
            self.startTime = time.time()
            cx, cy = currentHead

            imgR = np.zeros_like(img)
            imgR = cv2.circle(imgR, (self.opt.imgWidth//2, self.opt.imgHeight//2),
                            40, (255, 255, 255), 100)

            img = cv2.addWeighted(img, 0.7, imgR, 0.3, 1.0)

            if self.opt.imgWidth//2 - 40 < cx < self.opt.imgWidth//2 + 40 and \
                self.opt.imgHeight//2 - 40 < cy < self.opt.imgHeight//2 + 40:
                if self.waitCount<1:
                    self.playing = True
                else:
                    font = cv2.FONT_HERSHEY_COMPLEX
                    text = f"{self.waitCount}"
                    textsize = cv2.getTextSize(text, font, 3, 9)[0]
                    img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 550), font, 3, (0, 0, 0), 9)
                    textsize = cv2.getTextSize(text, font, 3, 5)[0]
                    img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 550), font, 3, (0, 0, 200), 5)

                    self.waitPlaying -= 1
                    if self.waitPlaying == 0:
                        self.waitPlaying = 10
                        self.waitCount -= 1

            else:
                font = cv2.FONT_HERSHEY_COMPLEX
                text = "Move your index finger"
                textsize = cv2.getTextSize(text, font, 2, 9)[0]
                img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 550), font, 2, (0, 0, 0), 9)
                textsize = cv2.getTextSize(text, font, 2, 5)[0]
                img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 550), font, 2, (0, 0, 200), 5)

                font = cv2.FONT_HERSHEY_COMPLEX
                text = "to the center of screen"
                textsize = cv2.getTextSize(text, font, 2, 9)[0]
                img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 650), font, 2, (0, 0, 0), 9)
                textsize = cv2.getTextSize(text, font, 2, 5)[0]
                img = cv2.putText(img, text, (int((img.shape[1] - textsize[0]) / 2), 650), font, 2, (0, 0, 200), 5)

                self.waitPlaying = 10
                self.waitCount = 3

        img[40:200,-200:-40] = logo

        return img

    def updateInGameScreen(self, imgMain):
        # Draw Food
        rx, ry = self.foodSPoint
        lx, ly = self.foodLPoint
        
        imgMain = cvzone.overlayPNG(imgMain, self.imgSFood, (rx-self.opt.SFoodSize[0]//2, ry-self.opt.SFoodSize[1]//2))
        if self.LFoodExist:
            imgMain = cvzone.overlayPNG(imgMain, self.imgLFood, (lx-self.opt.LFoodSize[0]//2, ly-self.opt.LFoodSize[1]//2))

        imgMain = cv2.rectangle(imgMain, (0,0), (800, 120), (0, 180, 0), -1)
        imgMain = cv2.putText(imgMain, self.opt.name, (20, 60), cv2.FONT_HERSHEY_COMPLEX,
                            2, (255, 255, 255), 3, cv2.LINE_AA)
        imgMain = cv2.putText(imgMain, f"Score: {self.score}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)
        if self.dying == 0:
            imgMain = cv2.putText(imgMain, f"Time: {time.time() - self.startTime:.2f}s", (500, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)
            self.endTime = time.time() - self.startTime
        else:
            imgMain = cv2.putText(imgMain, f"Time: {self.endTime:.2f}s", (500, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2, cv2.LINE_AA)

        imgMain = cv2.rectangle(imgMain, (5, 685), (600, 715), (0, 220, 0), 1)

        if self.LFoodExist:
            imgMain = cv2.rectangle(imgMain, (8, 688), (int((597 - 8) * self.LFoodWait / 30 + 8), 712), (0, 220, 0), -1)

        return imgMain

    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        if len(currentHead) == 0:
            cx, cy = self.previousHead
        else:
            cx, cy = currentHead
        # print(self.dying)
        
        if self.dying == 0:
            
            distance = math.hypot(cx-px, cy-py)
            print(distance)
            if distance > 5:
                self.points.append([cx, cy])

                self.lengths.append(distance)
                self.currentLength += distance
                self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)

                    if self.currentLength <= self.allowedLength:
                        break
        elif self.dying == 1:
            self.dying = 0
            self.gameOver = True
            self.inGame = False
        else:
            self.dying -= 1

        # Check if snake ate the food
        rx, ry = self.foodSPoint
        if rx - self.opt.SFoodSize[0]//2 < cx < rx + self.opt.SFoodSize[0]//2 and \
                ry - self.opt.SFoodSize[1]//2 < cy < ry + self.opt.SFoodSize[1]//2:
            self.randomSFoodLocation()
            self.allowedLength += self.lengthPerSFood
            self.score += self.scorePerSFood
            self.LFoodCount -= 1
            if self.LFoodCount == 0:
                self.LFoodExist = True

        lx, ly = self.foodLPoint
        # print(rx, ry, lx, ly)
        if self.LFoodExist:
            if lx - self.opt.LFoodSize[0]//2 < cx < lx + self.opt.LFoodSize[0]//2 and \
                    ly - self.opt.LFoodSize[1]//2 < cy < ly + self.opt.LFoodSize[1]//2:
                self.randomLFoodLocation()
                self.allowedLength += self.lengthPerLFood
                self.score += int(self.scorePerLFood * self.LFoodWait / 30)
                self.LFoodExist = False
                self.LFoodWait = 30
                self.LFoodCount = 5
            else:
                self.LFoodWait -= 1
                if self.LFoodWait < 0:
                    self.LFoodExist = False
                    self.LFoodWait = 30
                    self.LFoodCount = 5


        # Draw snake
        if self.points:
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(imgMain, self.points[i-1], self.points[i], (0,0,255), 20)
            cv2.circle(imgMain, self.points[-1], 20, (200,0,200), cv2.FILLED)

        # Check for Collision
        pts = np.array(self.points[:-2], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(imgMain, [pts], False, (0, 255, 0), 3)
        minDist = cv2.pointPolygonTest(pts, (cx, cy), True,)
        # if self.dying == 0:
        #     print(cx, cy)

        if (-1 <= minDist <= 1 or \
                300 > cx or cx > 980 or \
                200 > cy or cy > 520) and self.dying==0 and self.inGame:
            self.dying = 20
            self.saveData()

        return imgMain

    def saveData(self):
        option = 'play' if self.startGame else 'test'
        self.dataDict['HoTen'].append(self.opt.playerName)
        self.dataDict['Khoa'].append(self.opt.playerYear)
        self.dataDict['Nganh'].append(self.opt.playerMajor)
        self.dataDict['MSSV'].append(self.opt.MSSV)
        self.dataDict['Score'].append(self.score)
        self.dataDict['Option'].append(option)


        df = pd.DataFrame(self.dataDict,)
        df.to_csv(self.opt.main_csv_file, index=False)

        self.dataIndividualDict['HoTen'].append(self.opt.playerName)
        self.dataIndividualDict['Khoa'].append(self.opt.playerYear)
        self.dataIndividualDict['Nganh'].append(self.opt.playerMajor)
        self.dataIndividualDict['MSSV'].append(self.opt.MSSV)
        self.dataIndividualDict['Score'].append(self.score)
        self.dataIndividualDict['Option'].append(option)
        df = pd.DataFrame(self.dataIndividualDict,)
        os.makedirs('data/individuals', exist_ok=True)
        save_file = f"data/individuals/{opt.playerMajor}_{opt.playerYear}_{opt.playerName}_{opt.MSSV}.csv"
        df.to_csv(save_file, index=False)


def main(opt):
    game = SnakeGameClass(opt)

    cap = cv2.VideoCapture(0)
    cap.set(3,opt.imgWidth)
    cap.set(4,opt.imgHeight)
    cv2.namedWindow(opt.name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(opt.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    detector = HandDetector(detectionCon=0.6, maxHands=1)

    while True:
        success, img = cap.read()

        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)
        # print(hands)

        if game.menuScreen:
            img = game.getScreen()
            if hands:
                lmList = hands[0]['lmList']
                points = []
                points.append(lmList[8][0:2])
                points.append(lmList[12][0:2])
                cv2.circle(img, points[0], 20, (200,0,200), cv2.FILLED)
                cv2.circle(img, points[1], 20, (200,0,200))
                game.updateScreen(img, points)
        elif game.inGame:
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]
                img = game.getInGame(pointIndex)
                cv2.circle(img, pointIndex, 20, (200,0,200), cv2.FILLED)
                if game.playing:
                    img = game.update(img, pointIndex)
            elif game.dying != 0:
                img = game.getInGame([0,0])
                if game.playing:
                    img = game.update(img, [0,0])
            else:
                img = game.getInGame([0,0])
            img = game.updateInGameScreen(img)
        elif game.gameOver:
            img = game.getGameOver()
            if hands:
                lmList = hands[0]['lmList']
                points = []
                points.append(lmList[8][0:2])
                points.append(lmList[12][0:2])
                cv2.circle(img, points[0], 20, (200,0,200), cv2.FILLED)
                cv2.circle(img, points[1], 20, (200,0,200))
                game.updateGameOver(img, points)
        else:
            raise
        cv2.imshow(opt.name, img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Args parser for game options")
    parser.add_argument("--name", default="Super Snake XYZ3000", help="game name")
    parser.add_argument("--pathSFood", default="GreenApple.png", help="path for small food")
    parser.add_argument("--SFoodSize", default=[50,50], type=int, nargs="+", help="size of small food")
    parser.add_argument("--pathLFood", default="RedApple.png", help="path for large food")
    parser.add_argument("--LFoodSize", default=[80,80], type=int, nargs="+", help="size of small food")

    parser.add_argument("--imgHeight", default=720, type=int, help="image height")
    parser.add_argument("--imgWidth", default=1280, type=int, help="image width")

    ## Player Information
    parser.add_argument("--playerName", default="Do Quang Manh", help="Player Name")
    parser.add_argument("--playerYear", default="K15")
    parser.add_argument("--playerMajor", default="AI", type=str)
    parser.add_argument("--MSSV", default="HE153129", type=str)

    ## Logo
    parser.add_argument("--pathLogo", default="fds ava.png")

    ## Menu Screen options
    parser.add_argument("--pathScreen", default="screen.jpg", help="path for screen")

    ## In Game Screen options
    parser.add_argument("--pathInGameBackground", default="inGameB.jpg")
    parser.add_argument("--pathInGameBoundary", default="inGameBoundary.jpg")
    parser.add_argument("--pathInGameBoard", default="inGameBoard.jpg")

    ## Game Over options
    parser.add_argument("--pathGameOver", default="screen.jpg", help="path for game over screen")

    parser.add_argument("--main_csv_file", default="data/dataTrack.csv")

    opt = parser.parse_args()
    main(opt)