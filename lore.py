'''
Author: Nyuutrino
Date: 2025-2-22
Description: Displays the lore of the game in a star-wars like fashion.
Required libraries: pyphen
'''
from time import sleep
from LED import *
import pyphen

# Set up pyphen
pyphen.language_fallback("en_US")
dic = pyphen.Pyphen(lang='en_US')


# Displays the lore of the game 
def lore(delta):
	textcolor = (242, 225, 111)
	offset = -(delta / 15)
	refresh(color=(19, 35, 74))
	# Grab lore from file and store it in a string
	file = open("lore.txt", "r")
	lore = file.read()
	file.close()

	set_font(FNT_SMALL)

	# Strip & draw lines
	loreNewArr = lore.split("\n")
	loreArr = []
	for nL in loreNewArr:
		loreArr += nL.split(" ")
	loreArr = list(map(lambda s: s.replace("\n", ""), loreArr))
	loreLine = ""
	lettersTot = 15
	lettersUsed = 0
	lineNum = 0
	for word in loreArr:
		# Add word to line 
		if (lettersUsed + len(word) <= lettersTot):
			loreLine += word + " "
			lettersUsed += len(word)
			continue
		# Too many letters in the word. Split up by syllable or if it has a hyphen
		if (len(word) > lettersTot):
			tmpWord = word.split("-")
			if len(tmpWord) < 1:
				# No hyphen, split up by syllable
				tmpWord = dic.wrap(tmpWord, lettersTot)
			for breakup in tmpWord:
					lineNum += 1
					draw_text(0, lineNum * 10 + offset, breakup + "-", textcolor)
			loreLine = ""
			lettersUsed = 0
		# Too many letters in the line. Start a new line
		else:# Too many letters, draw a new line
			lineNum += 1
			draw_text(0, lineNum * 10 + offset, loreLine, textcolor)
			loreLine = word + " "
			lettersUsed = len(word)
	draw()

# Example
#set_orientation(1)
#delt = 0
#while True:
#	lore(delt)
#	delt += 1