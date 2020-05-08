# program to set up Spyfall game frame with bot player optionality, all players play taking turns on the command line

import random
import time
import signal
import tkinter
from tkinter import messagebox


# ask for players on the command line
def solicitPlayers():
    splayers = ""
    while splayers == "":
        splayers = input("Please enter the names of the human players, separated by a whitespace. \n")
        if splayers == "":
            print("You did not enter any names. \n")
    return splayers


def revealSpy(player, youspy, locpro):
    root = tkinter.Tk()
    root.withdraw()

    message = "Player " + player + ", please press OK for the reveal."
    if messagebox.showinfo("Reveal", message):
        if youspy:
            txt = "Spy"
        else:
            txt = "You are not a spy. The prompt for the location is: \"" + locpro + "\"."
        messagebox.showinfo("Reveal", txt)
        return

    root.mainloop()


def guess(locprompts, locpro):
    print("Hello Spy. I hope you're confident you've made the right decision. Here are the location prompts.\n")
    count = 0

    temp_locprompts = locprompts.copy() # so players dont catch on that order matches rounds
    random.shuffle(temp_locprompts)
    for l in temp_locprompts:
        print(count, ": ", l, "\n")
        count += 1
    gR = input("Enter the index of your predicted location prompt.\n")
    g = int(gR)
    if temp_locprompts[g] == locpro:
        print("The spy has guessed the correct location!\n")
        return True
    else:
        print("The spy has guessed incorrectly.\n")
        return False


def accuse(players):
    accused = ""
    while accused not in players:
        accused = input("Please enter the name of the person you would like to accuse: \n")
        if accused not in players:
            print("The accused is not one of the players. Try again.\n")
    pos_votes = 0
    for i in range(len(players)):
        if players[i] == accused: continue # skip over accused
        if players[i] in humans:
            vote = ""
            while (vote != "n") and (vote != "y"):
                vote = input("Player " + players[i] +": Will you accuse " + accused + "? (y/n)\n")
                if (vote != "n") and (vote != "y"): print("Please enter correct input.")
            if vote == "y": pos_votes += 1
        else:
            """@ED: IF THE BOT IS A SPY JUST SAY YES AKA INCREMENT POS_VOTES.
            IF NOT SPY CALL ON BOT TO MAKE AN ACCUSATION: IF THE ACCUSATION IS THE SAME AS accused THEN
            INCREMENT POS_VOTES. OTHERWISE DO NOTHING."""

    if (pos_votes * 2) > (len(players)-1): # again count out one player because they're accused
        return accused
    else:
        return None


def timeout_handler(signal, frame):
    raise Exception('Time is up!\n')


# remember to return who the spy was this time so they can be dealer next round
def round(locprompts, locpro, players, humans):
    # collects the answers of non-spy characters in case spy is a bot.
    # recreates basis for analysis that a human spy would have: others comments
    transcript = ""

    """@ED: THIS WAS ONE OF THE THINGS I DIDN'T UNDERSTAND. I DON'T KNOW WHAT INFO LIAM'S BOT
    USES TO FIND SOMEONE TO ACCUSE SO I DON'T KNOW IF THIS IS NECC. OR NEEDS TO BE REPLACED."""
    # transcript for each player so non-spy bots can analyze which player is the spy.
    # I figured the bot would need this to calculate probabilities and accuse.
    pscripts = []
    for i in range(len(players)):
        pscripts.append("")

    # randomize turn sequence
    random.shuffle(players)

    # choose spy randomly - spy is the index of the spy in players
    spy = random.randrange(len(players))

    """@ED: BOTS ARE INITIALIZED HERE IN PARTICULAR WAYS BASED ON SPY/NONSPY STATUS. THESE 
    PARTICULAR WAYS ARE DEFINED IN THE CONSTRUCTOR FOR THE BOT.  USE THE NONINTERSECTIONS B/W PLAYERS 
    AND HUMANS TO FIND BOTS. """

    time.sleep(0.25)
    # demonstrate turns and give instructions for next step
    print("\n FYI, the following is the turn sequence, including our bot players! You don't need to memorize " +
          "this.")
    for i in range(len(players)):
        print((i + 1), "-", players[i], end="\n")

    # when players hit enter, take them to be shown their status
    print("\n You will be taken to know your spy status or to be informed of the location one at a time. You will be " +
          "prompted based on your player name. There will be one reveal per player, so pay close attention!")
    for k in humans:
        input("Press enter to continue.")
        revealSpy(k, (k == players[spy]), locpro)

    print("\n _____GAME_START_____\n\n Here are the rules:\n The 8 minute timer will begin when you press enter. If you "
          "would like a running countdown, we recommend you to start your own visible timer exactly when you hit enter."
          "\n\n", players[0], "will start off the round. Each player will make a statement relevant to the location.")
    input("\n You can accuse players by typing 'ACCUSE' instead of a comment or end the round and guess the "+
          "location by typing 'GUESS'.")


    # P L A Y   T H E   G A M E

    signal.signal(signal.SIGALRM, timeout_handler)

    round_over = False
    signal.alarm(15)
    try:
        while not round_over:
            for k in range(len(players)):
                if players[k] not in humans:
                    # temporary
                    print("BOT GUESSES OR GENERATES")
                    '''@ED: 
                    IF bot is a spy: call on it to guess (you may need to provide it with the location prompts
                    as options, if so pass that as a parameter to round()) - if it is above a confidence level 
                    (we can try 0.75) it will return a guess (***make sure to also cancel alarm w/ signal.alarm(0)
                    , set win_spy (boolean did spy win) and set round_over to true and break)
                    REFERENCE GUESS() ABOVE- if its not  above the confidence level, it will return None,
                    so you just default to generating text, add to transcript, print comment
                    IF bot is not a spy: just generate text, add to transcript, print comment'''
                else:
                    comment = input("Player " + players[k] + ", please provide a comment: \n")
                    spltcom = comment.split()
                    first = spltcom[0]
                    if first == "GUESS":
                        if k != spy:
                            print("That was a bad idea. \n")
                            continue # goes to next players turn
                        signal.alarm(0) #cancels alarm
                        # handles the end of the game in this case
                        win_spy = guess(locprompts, locpro)
                        round_over = True
                        break

                    if first != "ACCUSE":
                        transcript.join(" " + comment)
                        players[k].join(" " + comment)
                    else:
                        left = signal.alarm(0)
                        print("Paused. Seconds left:", left, ".\n Reminder that these are the players.")
                        for i in range(len(players)):
                            print((i + 1), "-", players[i], end="\n")
                        print("\n")
                        accused = accuse(players)
                        if accused is not None:
                            if accused == players[spy]: return False, spy
                            else: return True, spy
                        input("The accusation lacked the necessary majority for a conviction. When "
                              "you are ready to resume your timer, press enter to continue the game.")
                        signal.alarm(left)
        return win_spy, spy
    except:  # only happens if game ends with end of timer, aka round needs to wrap up with accusations
        if not round_over:
            print("\n TIMES UP! TIME FOR FINAL ACCUSATIONS\n")
            for p in humans:
                print("Player "+ p + ":\n")
                accused = accuse(players)
                if accused is not None:
                    if accused == players[spy]:
                        return False, spy
                    else:
                        return True, spy
                print("The accusation lacked the necessary majority for a conviction.\n")
            return True, spy  # if they fail to convict, spy wins


if __name__ == "__main__":

    """@ED: PLEASE CHANGE this to a list of prompts instead of location names. Obv this is a lot of 
    locations so downsize as you deem necessary """

    locprompts = ["Airplane", "Bank", "Beach", "Cathedral", "Circus Tent", "Corporate Party", "Crusader Army",
                 "Casino", "Day Spa", "Embassy", "Hospital", "Hotel", "Military Base", "Movie Studio", "Ocean Liner",
                 "Passenger Train", "Pirate Ship", "Polar Station", "Police Station", "Restaurant", "School",
                 "Service Station", "Space Station", "Submarine", "Supermarket", "Theater", "University",
                 "World War II Squad"]
    players = []

    print("Welcome to Spyfall! You can play with humans and up to 3 bots.")
    # ask for players and correct for common mistakes
    while not players:
        players = solicitPlayers().split(' ')
        i = 0
        while i != len(players):
            if players[i] == "":
                players.pop(i)
                i -= 1
            i += 1
        if not players:
            print("You did not enter any names. \n")

    # ask for number of bots
    botNum = 4
    while botNum > 3 or botNum < 0:
        botNumR = input("\nNow please enter how many bots you would like to play the game: up to 3 \n")
        try:
            botNum = int(botNumR)
        except ValueError:
            print("Not an integer input.\n")
            continue
        if botNum > 3 or botNum < 0:
            print("The number is not in the specified range. \n")

    # will need this later
    humans = players.copy()

    # add bots to player list
    bNames = ["bSun", "bCloud", "bRain"]
    for i in range(botNum):
        players.append(bNames[i])

    leaderboard = {}  # dictionary that will keep track of points for the game
    for p in players:
        leaderboard[p] = 0

    rounds = 0
    while (rounds < 2) or (rounds > 7):
        roundsR = input("\n How many rounds would you like to play ? 2-7 (5 is recommended)")
        try:
            rounds = int(roundsR)
        except ValueError:
            print("Not an integer input.\n")
            continue
        if (rounds < 2) or (rounds > 7): print("Out of range")


    roundLocs = []
    # randomly assign locations to each round
    random.shuffle(locprompts)
    for j in range(rounds):
        roundLocs.append(locprompts[j])

    for r in range(rounds):
        print("\n______ROUND ", (r+1), "______")
        spy_win, spy = round(locprompts, roundLocs[r], players, humans)
        if not spy_win:
            print("The non-spies have won the round! Each of them is awarded two points.")
            for i in range(len(players)):
                if i == spy: continue
                leaderboard[players[i]] += 2  # used dictionary because players randomizes order each round
        else:
            print("The spy "+ players[spy] + " has won the round! They are awarded two points. \n")
            leaderboard[players[spy]] += 2
        time.sleep(2)

    max_points = 0
    winner = ""
    for key_player in leaderboard:
        if leaderboard[key_player] > max_points:
            max_points = leaderboard[key_player]
            winner = key_player

    winners = []
    for key_player in leaderboard:
        if leaderboard[key_player] == max_points:
            winners.append(key_player)

    print("THE WINNERS OF THIS SPYFALL GAME ARE ")
    for w in winners:
        print(w)
    print("CONGRATULATIONS!")