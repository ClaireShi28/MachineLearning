# Design Information
### 1. When the application is started, the player may choose to (1) Play a word game, (2) View statistics, or (3) Adjust the game settings.  

This requirement is realized by creating a 'Player' class to track the user using a string 'name' and operates a start() 
method to initialize game settings (letter pool, maximum number of turns). A 'LetterPool' class is added to record initial 
letter pool setting using a List 'letterPool' and a Map 'pointEachLetter'. 

### 2. When choosing to adjust the game settings, the player (1) may choose for the game to end after a certain maximum number of turns and (2) may adjust the number of and the letter points for each letter available in the pool of letters, starting with the default matching the English Scrabble distribution (12 E’s worth 1 point each, 4 D’s worth 2 points each, etc).

This requirement is realized by creating a 'GameSettings' class to set things. I used (1) an Integer to record 'maxTurns', (2) a 
Map to record the 'numberEachLetter', (3) another Map to record the 'pointEachLetter'. 

### 3. When choosing to play a word game, a player will:
### a. Be shown a ‘board’ consisting of 4 letters drawn from the pool of available letters.

A 'Board' class is added to realized this requirement. An Integer 'boardSize' set to 4 and the List 'letterPool' are used 
as inputs for the draw(Integer, List) method, which returns the current letters on the board using a List 'board'. The 'letterPool' 
is then updated by method removeLetterPool(List: board), which removes letters selected for 'board' from the pool.

### b.Be shown a ‘rack’ of 7 letters also drawn from the pool of available letters.

A 'Rack' class is added to realized this requirement. An Integer 'rackSize' set to 7 and the List 'letterPool' are used 
as inputs for the draw(Integer, List) method, which returns the current letters in the rack using a List 'tiles'. The 
'letterPool' is also updated by method removeLetterPool(List: tiles), which removes letters selected for 'tiles' from the pool.

### c. On each turn, up to the maximum number of turns either: 
#### i. Enter a word made up of one or more letters from the player’s rack of letters, and one from the board. The word must contain only valid letters and may not be repeated within a single game, but does not need to be checked against a dictionary (for simplicity, we will assume the player enters only real words that are spelled correctly).or

To realize this requirement, I added a class 'Word'. The player can then enters the word which is recorded in String 'word'. 

By getRackWord() method, the List 'rack_word' in 'Rack' is used to track the letters in the 'word' that are from the player's rack. 
By getBoardLetter() method, the String 'board_word' in 'Board' tracks the letter in the 'word' from the board.

In 'Word', the List 'wordGame' is used to record all the valid words played in this game.

To make sure the word that player entered is not repeated in this game, method checkWord() is used to see if 'word' exists in 'wordGame', 
if result if true then throw warning message, else update 'wordGame' List, calculate the 'wordScore' Integer by getScore() method. 

The number of turns is updated using Integer 'turns' and method updateTurn() which returns 'turns'++.

#### ii.Swap 1-7 letters from their rack with letters from the pool of letters.  This is the only time letters are returned to the pool during a game.

If swap operation chosen, the letters in 'tiles' is/are returned into letter pool by method returnLetterPool(Integer: 1-7).  
The letters in 'tiles' are updated by method swap(Integer: 1-7, List) which randomly select letters (depends on how many swapped) from
'letterPool' List. The 'letterPool' is then updated using removeLetterPool() method.
The number of turns is updated using Integer 'turns' and method updateTurn() which returns 'turns'++.

### d. After a word is played, that letter on the board will be replaced by a different random letter from the word that was just played.  Example:  If ‘c’ is on the board, and ‘j’,’a’,’k’,’e’,’t’,’s’ are part of the rack, then the player may enter ‘jackets’ as a word, and the ‘c’ will be randomly replaced by the ‘j’,’a’,’k’,’e’,’t’, or ’s’ for the next turn on the board.

This requirement is realized by method replace(String: board_word, List: rack_word) in 'Board', which updates the 'board'
List. The String 'board_word' records the letter (in the word played) that comes from 'board' by method getBoardLetter(). 
The List 'rack_word' records letters (played) from 'rack' by method getRackWord(). The String 'board_word' in the List 'board'
is randomly replaced by a letter from List 'rack_word'.

### e. After a word is played, the tiles used from the rack are replaced from the pool of letters.

This is realized by methods updateRackSize(List: rack_word) and refill(Integer: rackSize, List: letterPool). The former 
method update Integer 'rackSize' by subtracting the length of 'rack_word' then use refill method to refill correct number
of letters from pool to 'tiles'.

### f. After a word is played, the player’s score will increase by the total number of points for the letters in the word, including the letter used from the board. (So ‘jackets’, if using default values, would score 20 points.)

This is realized by first using the getScore(String: word, Map: pointEachLetter) method to calculate the word score 
Integer 'wordScore'. Then the total score for current game is updated using Integer 'totalScore' and method 
sumTotalScore(Integer: wordScore).

### g. If the pool of letters is empty and the rack cannot be refilled, the player will score an additional 10 points.

The method poolEmpty(List) in 'LetterPool' method rack_refillable() in 'Rack' are used to check if the player can get 
additional 10 points. If both true, then the Integer 'totalScore' is update by sumTotalScore(Integer: 10).

### h. When the maximum number of turns has been played, or the pool of letters is empty and the rack cannot be refilled, the game will end, and the final score will be displayed before returning to the first menu.

The method maxPlayed() is used to check is 'turns' has reached 'maxTurns', if true then end game and method gameEnded() 
returns true. Or if poolEmpty(List) in 'LetterPool' and rack_refillable() in 'Rack' both return true, the game ends as well.
Under the condition that gameEnded() returns true, method displayFinalScore(Integer: totalScore) is used to show final score.
The method reset() is used to reset all related variables to the setting of a new game.

### 4. A player may choose to leave a game in progress at any time.  Selecting to play a game from the menu should then return to the game in progress.

This requirement is realized by leaveGame() method. When select to play game next time, the gameEnded() method returns false
and last unfinished game's settings and variables are unchanged.

### 5. When choosing to view statistics, the player may view (1) game score statistics, (2) letter statistics or (3) the word bank.

I added Class 'Statistics' to the design. Classes 'GameScoreStatistics', 'WordBank' and 'LetterStatistics' are added as 3 types of
'Statistics'.

### 6. For game score statistics, the player will view the list of scores, in descending order by final game score, displaying:
### a. The final game score; 
### b.The number of turns in that game; 

These are realized by method updateScoreStats() in 'Game' and method scoreAverage() in 'GameScoreStatistics'. The method
updateScoreStats() is implemented each time after the game ends and before reset, it will return a Map 'scoreStats' with information 
of 'gameID', 'totalScore', 'turns'. 

### c.The average score per turn

The scoreAverage() will calculate average score and update 'scoreStats'. displayScoreStats() method
displays 'scoreStats' based on descending order of 'totalScore'.

### The player may select any of the game scores to view the settings for that game’s maximum number of turns, letter distribution, and letter points.
The settings() method in 'GameSettings' records each 'gameID' and corresponding information of 'maxTurns', 'numberEachLetter' and 'pointEachLetter'.
The method displaySettings() in 'GameScoreStatistics' is used to retrieve game settings information based on which 'gameID' selected. 

### 7. For letter statistics, the player will view the list of letters, in ascending order by number of times played, displaying:
### a. The total number of times that letter has been played in a word
### b. The total number of times that letter has been traded back into the pool
### c. The percentage of times that the letter is used in a word, out of the total number of times it has been drawn

Each time the method swap() in 'Rack' is implied or checkWord() method in 'Word' returns true, the Map 'letterStats' that recording the statistics
information of letters is updated by updateSwapStats() in 'Rack' and updateLetterStats() in 'Word' respectively. The percentage is then calculated
by method percentage() in 'LetterStatistics' and then updates 'letterStats'. The 'letterStats' is displayed by method displayLetterStatistics(), 
which ranks results in ascending order based on number of times played.

### 8. For the word bank, the player will view the list of words used, starting from the most recently played, displaying:
#### i. The word
#### ii. The number of times the word has been played

Each time a word is played, the Date 'word_time' and Integer 'word_number' are updated and then recorded in the Map 'word_bank' by method updateWordBank() in 'Word'. 
The Map 'word_bank' includes the information of all words played and total times each word has been played in all games. The 'word_bank' Map is then displayed by method displayWordBank() starting from most recent 'word_time'.

### 9. The user interface must be intuitive and responsive.
This is not represented in my design, as it will be handled entirely within the GUI implementation.

### 10. The performance of the game should be such that students do not experience any considerable lag between their actions and the response of the application.
This is not represented in my design, as it will be handled entirely within the GUI implementation.

### 11. For simplicity, you may assume there is a single system running the application.

