from flask import Flask, render_template, request, redirect, url_for, session, flash
import requests
import numpy as np
import pandas as pd
import joblib 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'temp'

# model loaders
MODELPATH = 'static/modeldata/model.joblib'
SCALPATH = 'static/modeldata/scaler.joblib'
model, scaler = joblib.load(MODELPATH), joblib.load(SCALPATH)

#Function to return the history API
def getHISTORY(fpl_id):
    # A get request to the API, this is familiar throughout.
    history_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/")
    # parse and return
    history_data = history_response.json()
    return history_data.get('current', [])

def rankCHANGEpercent(gameweeks):
    if len(gameweeks) >= 2:

        # second to last gameweek
        secondgwbefore = gameweeks[-2]
        #the gameweek before
        gwbefore = gameweeks[-1]

        #overall_rank is a variable in the API
        secondlastrank = secondgwbefore['overall_rank']
        lastrank = gwbefore['overall_rank']
        

        # percent calculator
        finalRANKPERCENT = ((secondlastrank - lastrank) / secondlastrank) * 100
        return finalRANKPERCENT
    else:
        return 0

def freeTRANSFERcalc(fpl_id):

    #same API request structure
    history_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/")

    history_data = history_response.json()

    # these are consistent variables across the API
    thisSeason = history_data.get('current', [])
    free_transfers = 1  # Start with 1 ft


    # this is a structure uses to work out the amount of free transfers.
    # there is no single metric to work this out, so we have to iterate through history to find out
    for gameweek in thisSeason:
        event_transfers = gameweek.get('event_transfers', 0)

    # If no transfers are made in the last event, add free transfers
        if event_transfers == 0:
                #until max of 2 transfers (cause that's the max in FPL)
                free_transfers = min(free_transfers + 1, 2)
        elif event_transfers == 1:
            if free_transfers < 2:
                free_transfers = 1
        else: 
            free_transfers = 1

    return free_transfers

def playerINFO2(fpl_id):
    league_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/") #api call as usual, but for fpl_id which is the entry
    league_data = league_response.json() # that league data is a varibale in the API

    player_last_name = league_data.get('player_last_name', 'Not available') # I will still use this
    player_first_name = league_data.get('player_first_name', 'Not available') #same variable


    #get the flag from the country they are from
    whichFlag = league_data.get('player_region_iso_code_short', 'Not available').lower()
    team_name = league_data.get('name', 'Not available')


    #bank and value (cost part)
    bank = league_data.get('last_deadline_bank', 'Not available')
    value = league_data.get('last_deadline_value', 'Not available')


    player_name = player_first_name + " " + player_last_name #combines the player name (as they're separate variables)
    return player_name, whichFlag, team_name, bank, value

# sec chance rank bit
def rank2Chance(fpl_id):
    response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/")

    RANKdata = response.json()
    # it's in the classic leagues bit
    #my issue here is there is no history, so I can't work out percent change. Will likely leave it in. 
    LEAGUEdata = RANKdata['leagues']['classic']
    for league in LEAGUEdata:
        if league['name'] == 'Second Chance':
            return league.get('entry_rank', 'Not available')
    return None


def chipcheckers(fpl_id): #to get the chips used
    history_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/")
    history_data = history_response.json()

    chips = history_data.get('chips', [])
    chipcounter = {}
    for CHIP in chips:


        chip_name = CHIP['name']
        chipcounter[chip_name] = chipcounter.get(chip_name, 0) + 1

    chip_status = {
        # Determines the CSS styling of each chip
        'wildcard': 'Yes' if chipcounter.get('wildcard', 0) >= 2 else 'No',
        'bboost': 'Yes' if chipcounter.get('bboost', 0) > 0 else 'No',
        'freehit': 'Yes' if chipcounter.get('freehit', 0) > 0 else 'No',
        '3xc': 'Yes' if chipcounter.get('3xc', 0) > 0 else 'No'
    }

    return chip_status
def BESTrank1(fpl_id):
    history_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/")
    # similar structure
    history_data = history_response.json()


    pastSeasons = history_data.get('past', []) #past & current seasons
    thisSeason = history_data.get('current', [])
    pastSeasons = history_data.get('past', [])

    #iterate through to get the best rank, print it on html
    currentHIGH = min([gw['overall_rank'] for gw in thisSeason if 'overall_rank' in gw], default="Not available")
    #toptest = max(['overall_rank'] / [overall_rank])
    oldHIGH = min([season['rank'] for season in pastSeasons if 'rank' in season], default="Not available")

    return currentHIGH, oldHIGH, #toptest

# function to sort the rank
def tL_sort(url, rank_sort):
    #
    response = requests.get(url)
    data = response.json()

    standings=0 #make numberical
    standings = data.get('standings', {}).get('results', [])

    return next((player.get('total') for player in standings if player.get('rank_sort') == rank_sort), "No data on this.") #return

def diff_2RANK(summary_overall_points):
    ranklink = {
        #finally worked it out, I had to match the API's with a specific person in this rank area
        "1m": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=20000",
        "100k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=2000",
        #test"110k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=2100",
        "10k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=200",
        "1k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=20",
        "1": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=1"
    }

    rank_differences = {}
    #will style the rank gradient in the my ai bit eventually
    CSSrank_styles = {}

    for LAB1X, url in ranklink.items():
        rank_sort = int(LAB1X.replace('k', '000').replace('m', '000000'))
        total_for_rank = tL_sort(url, rank_sort)
        if total_for_rank == "Data not found":
            difference = None
            temprankdifference = 0
            del temprankdifference
        else:
            difference = total_for_rank - summary_overall_points
        rank_differences[LAB1X] = difference
        # 



        # Assign CSS class based on difference
        if difference is None:
            CSSrank_styles[LAB1X] = 'rankstore'
        elif difference <= 0:
            CSSrank_styles[LAB1X] = 'rankstore'
        elif difference <= 15:
            CSSrank_styles[LAB1X] = 'rankstore90'
        elif difference <= 25:
            CSSrank_styles[LAB1X] = 'rankstore80'
        elif difference <= 40:
            CSSrank_styles[LAB1X] = 'rankstore70'
        elif difference <= 55:
            CSSrank_styles[LAB1X] = 'rankstore60'
        elif difference <= 75:
            CSSrank_styles[LAB1X] = 'rankstore50'
        elif difference <= 100:
            CSSrank_styles[LAB1X] = 'rankstore40'
        elif difference <= 130:
            CSSrank_styles[LAB1X] = 'rankstore30'
        elif difference <= 160:
            CSSrank_styles[LAB1X] = 'rankstore20'
        elif difference <= 190:
            CSSrank_styles[LAB1X] = 'rankstore10'
        else:
            CSSrank_styles[LAB1X] = 'rankstore0'

    return rank_differences, CSSrank_styles


def DATAFETCH(fpl_id): #just fetches the hisotry again
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# extract the rank data again (for myai)
def EXTRACTDAT(data, fpl_id):
    if data and 'current' in data:
        return [{
            'fpl_id': fpl_id,
            'event': event_data['event'],
            'overall_rank': event_data['overall_rank'],
            #'event_id'


        } for event_data in data['current']] #has to be CURRENT EVENT. this is important. This is set using the 32 variable later.
    else:
        return []

# For the rank prediction AI. 

#this is feature engineering, the added features are rank_change, rank_volatility etc. 
def COMPUTEIT(df):
    df['rank_change'] = df.groupby('fpl_id')['overall_rank'].diff().fillna(0)

    df['avg_rank_last_3'] = df.groupby('fpl_id')['overall_rank'].rolling(window=3).mean().reset_index(level=0, drop=True).fillna(0)

    df['rank_volatility'] = df.groupby('fpl_id')['overall_rank'].rolling(window=5).std().reset_index(level=0, drop=True).fillna(0)


    df['rank_momentum'] = df.groupby('fpl_id')['rank_change'].rolling(window=3).sum().reset_index(level=0, drop=True).fillna(0)

    df['relative_rank_change'] = df.groupby('fpl_id')['overall_rank'].pct_change().fillna(0)  # hard coded this.
    return df.dropna()

# prediction simulator, justification on noise explained.
def SIMPREDS(model, scaler, user_df, n_simulations=100, BASEnoise=0.05, bias2fac=0.7, WORST10OUT=10):
    X = user_df.drop(['overall_rank', 'fpl_id', 'event'], axis=1)
    X_scaled = scaler.transform(X)
    
    current_rank = user_df['overall_rank'].iloc[-1]  # Most recent overall rank, pred isn't allowed to be higher, checks it.
    predsvalue = len(user_df) * 0 

#noise adjusted for different ranks (higher ranks=lower noise), that's what tends to happen. 
    predictions = []
    print("\nIndividual Predictions:")
    for i in range(n_simulations):
        noise_std = BASEnoise
        if current_rank <= 10:

            noiseXX = 0.01 #adjust

        elif current_rank <= 1000:
            noiseXX = 0.1
        elif current_rank <= 10000:
            noiseXX = 0.5
        elif current_rank <= 100000:
            noiseXX = 1.2
        else:
            noiseXX = 2.5
        
        noise = np.random.normal(0, noise_std * noiseXX, X_scaled.shape)
        noisyDAT = X_scaled + 0 + noise
        # for pred. 
        pred = model.predict(noisyDAT)
        
        # slight bias towards people using the system as they're likely to be interested in the game.
        bitofBIAS = pred * bias2fac
        corrected_pred = max(round(bitofBIAS[-1]), 1)
        
        predictions.append(corrected_pred)
        print(f"Prediction {i + 1}: {corrected_pred}")

    #% exculsion (gets the worst 10 out of the average)
    exclude10 = int(len(predictions) * (WORST10OUT / 100))
    filtPred = sorted(predictions)[:-exclude10] if exclude10 > 0 else predictions

    avg_pred = np.mean(filtPred)
    best_pred = np.min(predictions)
    worst_pred = np.max(predictions)
    
    avg_pred = round(avg_pred)
    best_pred = round(best_pred)
    worst_pred = round(worst_pred) #*
    #median_pred = med(avg_pred)?

    BESTpred = min(best_pred, current_rank)

    THRESH = [1000000, 100000, 10000, 1000, 10] #rank places
    percentages = {threshold: (sum(pred < threshold for pred in predictions) / n_simulations) * 100 for threshold in THRESH}

    return avg_pred, BESTpred, worst_pred, percentages

# for the graph on pa
def last10GWRANK(fpl_id):
    #Return user history though the following API.  
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history/"
    response = requests.get(url)
    data = response.json()
    thisSeason = data['current']
    evTOTALs = len(thisSeason)
    #overall_rank data for mapped to each of the last 5 events
    if evTOTALs >= 10:
        LAST5events = thisSeason[-10:]
        events = [event['event'] for event in LAST5events]
        ranks = [event['overall_rank'] for event in LAST5events]
        #graph styling
        plt.figure(figsize=(24, 2), facecolor='black')
        plt.plot(events, ranks, marker='o', color='lightblue', linestyle='-', linewidth=5, 
                markerfacecolor='black', markeredgecolor='white', markeredgewidth=2)
            
        #turn axis off
        plt.gca().axis('off')
        plt.gca().set_ylim(max(ranks)+max(ranks)*0.1, min(ranks)-min(ranks)*0.1)
            
        for i, txt in enumerate(ranks):
            plt.annotate(txt, (events[i], ranks[i]), textcoords="offset points", xytext=(0,10), 
                         ha='center', color='white', fontsize=12)

        if not os.path.exists('static'):
            os.makedirs('static')
        #save to image path
        image_path = f'static/fpl_graph_{fpl_id}.png'
        plt.savefig(image_path, facecolor='black')
        plt.close()
            
        return image_path 
    
def favoteam(fpl_id):
    #this function gets the users favourite team and puts it in the kotgw link
    user_url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/"
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/" #map the id back to bootstrap
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    try:
        # Fetching user's favorite team
        userresp = requests.get(user_url)
        favteam = userresp.json().get('favourite_team')
        # Fetching static data for all teams
        bootresp = requests.get(bootstrap_url)
        bootstrap_data = bootresp.json()
        # Extract favourute team details
        FavteamDETAILS = next((team for team in bootstrap_data['teams'] if team['id'] == favteam), None)

        if FavteamDETAILS:
            # Fetching fixture data
            fixresp = requests.get(fixtures_url)
            fixresp.raise_for_status()
            fixtures_data = fixresp.json()
            eventNo = 38  #gameweek 32
            nextFIX = next((fixture for fixture in fixtures_data if (fixture['event'] == eventNo and (fixture['team_a'] == favteam or fixture['team_h'] == favteam))), None)

            # map the team and opponent
            # the API has team_a and team_h
            #Must determine which team is home and which is away from these variables. 
            if nextFIX:
                oppID = nextFIX['team_h'] if nextFIX['team_a'] == favteam else nextFIX['team_a']
                oppDets = next((team for team in bootstrap_data['teams'] if team['id'] == oppID), None)
                #the variables on the left are all from the boostrap API. I kept the variable names, and these are consistent throughout. 
                team_details = {
                    "favteam": favteam,
                    "name": FavteamDETAILS['name'],
                    #team strength metrics
                    "strength": FavteamDETAILS['strength'],
                    # these are mapped depending on team_h and team_a. 
                    "strength_overall_home": FavteamDETAILS['strength_overall_home'],
                    "strength_overall_away": FavteamDETAILS['strength_overall_away'],
                    "strength_attack_home": FavteamDETAILS['strength_attack_home'],
                    "strength_defence_home": FavteamDETAILS['strength_defence_home'],
                    "strength_attack_away": FavteamDETAILS['strength_attack_away'],
                    "strength_defence_away": FavteamDETAILS['strength_defence_away'],
                    "opponent_name": oppDets['name'] if oppDets else 'Unknown',
                    "opponent_id": oppID,
                }

                # Calculating total strengths
                team_details['total_strength_overall'] = team_details['strength_overall_home'] + team_details['strength_overall_away']
                team_details['total_strength_attack'] = team_details['strength_attack_home'] + team_details['strength_attack_away']
                team_details['total_strength_defence'] = team_details['strength_defence_home'] + team_details['strength_defence_away']

                # Finding the ranks for each team
                # and add them
                teams = [{
                    'name': team['name'],
                    'overall_strength': team['strength_overall_home'] + team['strength_overall_away'],

                    'attack_strength': team['strength_attack_home'] + team['strength_attack_away'],
                    'defence_strength': team['strength_defence_home'] + team['strength_defence_away'],
                    # test if field is available
                    'test_field': 'teams' if team['name'] or not team['name'] else 'not a field'
                } for team in bootstrap_data['teams']]
                overallSORTER = sorted(teams, key=lambda x: x['overall_strength'], reverse=True) # sort by strength for each metrics
                attackSORTER = sorted(teams, key=lambda x: x['attack_strength'], reverse=True)
                defenceSORTER = sorted(teams, key=lambda x: x['defence_strength'], reverse=True)

                team_details['overall_rank'] = next((index + 1 for index, team in enumerate(overallSORTER) if team['name'] == FavteamDETAILS['name']), None)
                team_details['attack_rank'] = next((index + 1 for index, team in enumerate(attackSORTER) if team['name'] == FavteamDETAILS['name']), None)
                team_details['defence_rank'] = next((index + 1 for index, team in enumerate(defenceSORTER) if team['name'] == FavteamDETAILS['name']), None)

                return team_details
            else:
                return {"error": "You have not picked a favourite team."}
        else:
            return {"error": "You have not picked a favourite team."}
    # error expection just in case. 
    except requests.RequestException as e:
        return {"Team ERROR!": str(e)}
    
# final link to the predicted dataset
# this will change every week, but for demo purposes we will use gameweek 32. 
predictionsFILEPATH = 'static/final_datasets/predicted_datasets/final_32.csv'
final_data = pd.read_csv(predictionsFILEPATH, index_col=0)

def playerPicks1(fpl_id):
    #gw32 player picks.
    url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/38/picks/"
    response = requests.get(url)

    return response.json()

def TOTALpredpoints(fpl_id):
    #load csv again
    final_data = pd.read_csv(predictionsFILEPATH, index_col=0)
    #similar usage
    def playerPicks1(fpl_id):
        url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/38/picks/"
        response = requests.get(url)

        return response.json() 

    fpl_data = playerPicks1(fpl_id)
    #set the predicted points to a number
    predictedNumberFINAL = 0
    if fpl_data:
        picks = fpl_data.get('picks', [])
        pick = []
        picks = fpl_data.get('picks', [])
        for pick in picks:
            player_id = pick['element'] #iterates over each element in the csv. 
            Prow1 =  final_data.loc[final_data['id'] == player_id]
            if not Prow1.empty: #just checks if it's there

                predicted_points = Prow1['predicted_total_points'].values[0]
                predictedNumberFINAL += predicted_points
        foundALLplayerS = len(picks)

        tempaddedpoints = CALCULATETHEtempaddedpoints(foundALLplayerS)
        # for players that are not in the dataset (when their value is negative)
        predictedNumberFINAL += tempaddedpoints

    return predictedNumberFINAL

def CALCULATETHEtempaddedpoints(num_players_found):
    missing_players = 15 - num_players_found #for missing elements
    tempaddedpoints = min(missing_players, 5) * 2
    return tempaddedpoints

def lowestplayerFind(fpl_data, final_data):
    #find the players with the lowest predicted points
    # mainly for transfer recs
    team_players = [pick['element'] for pick in fpl_data.get('picks', [])]

    team_data = final_data[final_data['id'].isin(team_players)]#

    lowestplayers = {'DEF': None, 'MID': None, 'FWD': None}
    #iterate over each position
    #don't need GK as you would never make that transfer
    #remove GK position
    for position in ['DEF', 'MID', 'FWD']:
        positionplayersXcolumn = team_data.query("position == @position")



        if not positionplayersXcolumn.empty: # simple error handling

            lowest_player = positionplayersXcolumn.nsmallest(1, 'predicted_total_points').iloc[0] #icloc selects the lwest player
            lowestplayers[position] = lowest_player

    return lowestplayers #ret



def findHighests(fpl_data, final_data):
    team_players = [pick['element'] for pick in fpl_data.get('picks', [])]
    team_data = final_data[final_data['id'].isin(team_players)] #
    highest1Player = {'DEF': None, 'MID': None, 'FWD': None}
    for position in ['DEF', 'MID', 'FWD']: #the same method
        
        positionplayersXcolumn = team_data[team_data['position'] == position]
        if not positionplayersXcolumn.empty:

            highest_player = positionplayersXcolumn.nlargest(1, 'predicted_total_points').iloc[0]


            highest1Player[position] = highest_player
    #final return
    return highest1Player

# hard coding the predicted goals. The predictions are based on expected goals csv. map the team id to the id below.
hardcodegoals = {
    1: 2, 2: 2, 3: 0, 4: 1, 5: 2, 6: 0, 7: 3, 8: 1,
    9: 0, 10: 1, 11: 1, 12: 1, 13: 3, 14: 1, 15: 1,
    16: 0, 17: 0, 18: 3, 19: 1, 20: 0 #represents goals
}

def best_playerINWHOLETEAM(fpl_data, final_data):

    highest1Player = findHighests(fpl_data, final_data)
    
    # Initalised variable
    highest_scoring_player = None
    highestplayerscore = -1

    
    # Go over each position, 
    for position, player in highest1Player.items():
        if player is not None and player['predicted_total_points'] > highestplayerscore:
            highestplayerscore = player['predicted_total_points']
            highest_scoring_player = pow
            highest_scoring_player = player
    
    # Highest scoring player gets doubled
    if highest_scoring_player is not None:
        highest_scoring_player = highest_scoring_player.copy()  # So the original data isn't changed.

        highest_scoring_player['x2predpoints'] = highest_scoring_player['predicted_total_points'] * 2
    
    return highest_scoring_player

def affordedp(lowestplayers, final_data, bank):
    player3AFFORD = {} #set dict
    positions = ['DEF', 'MID', 'FWD'] #removed GK bit


    for position in positions:  
        #emptyarray
        player3AFFORD[position] = []
        affCosT = lowestplayers    [position]['cost'] + bank
        player3AFFORD_list = final_data[(final_data['position'] == position) & (final_data['cost'] <= affCosT)].nlargest(3, 'predicted_total_points')#best player recommended

        for _, player in player3AFFORD_list.iterrows(): #pass through rows. 
            player3AFFORD[position].append({
                'id': player['id'], #array of id, name etc. 
                'name': player['name'],
                                'name': player['name'],
                'predicted_points': player['predicted_total_points'],
                'cost' : player['cost']#for value
            })
    return player3AFFORD

def predictedFAVTEAM(fpl_id):
    user_url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/"
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    try:
        userresp = requests.get(user_url)
        userresp.raise_for_status()
        favteamid = userresp.json().get('favourite_team')#getteam

        bootresp = requests.get(bootstrap_url)
        bootresp.raise_for_status()

        bootstrap_data = bootresp.json()

        team_players = [player for player in bootstrap_data['elements'] if player['team'] == favteamid]
        ledgewplayers = []#empty it
        for player in team_players:
            chance = player.get('chance_of_playing_next_round') #metric in the API
            form = float(player.get('form', 0)) #player form (another metric)
            if (player['element_type'] == 1 and chance is None) or \
               (chance is None and form >= 0.5) or \
               (74 <= (chance or 0) <= 100 and form >= 0.5):
                

                ledgewplayers.append(player)
        
        minSORT1 = sorted(ledgewplayers, key=lambda x: x['minutes'], reverse=True)

        goalkeepers = [player for player in minSORT1 if player['element_type'] == 1][:1]#sorted
        defenders = [player for player in minSORT1 if player['element_type'] == 2][:4]
        slotrem_ = 11 - len(goalkeepers) - len(defenders) #gk and def length

        topPLAYS = [player for player in minSORT1 if player['element_type'] not in [1, 2]][:slotrem_]
        topXI = goalkeepers + defenders + topPLAYS #adding all the 11
        #make sure you sort ellis
        elementTOP = sorted(topXI, key=lambda x: x['element_type'])#element type is 1-4

        return elementTOP
    except requests.RequestException as e: #except requester
        print(str(e))
        return []
    
def FullFAVTEAMplayers(favteam_id):
    # Read CSV
    data = pd.read_csv(predictionsFILEPATH, index_col=0)
    #match
    team_players = data[data['team'] == favteam_id]

    top3Players = team_players['name'].head(3).tolist()#parse to list
    return top3Players

#end of function area. 

#simple login area
#simple login area
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('players'))
    return render_template('login.html')


@app.route('/players', methods=['GET', 'POST'])
def players():
    if request.method == 'POST':
        fpl_id = request.form['fpl_id']
        session['fpl_id'] = fpl_id

        try:
            team_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/38/picks/")
            team_data = team_response.json()

            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
            data = response.json()

            player_id_to_details = {player['id']: (player['web_name'], player['team'], player['element_type']) for player in data['elements']}
            team_id_to_name = {team['id']: team['name'] for team in data['teams']}

            gw26_fixtures_response = requests.get("https://fantasy.premierleague.com/api/fixtures/")
            gw26_fixtures = [fixture for fixture in gw26_fixtures_response.json() if fixture['event'] == 38]

            # Mapping team ID to their Gameweek 26 opposing team, difficulty, and home/away status
            team_to_gw26_details = {}
            for fixture in gw26_fixtures:
                home_team = team_id_to_name[fixture['team_h']]
                away_team = team_id_to_name[fixture['team_a']]
                team_to_gw26_details[fixture['team_h']] = (away_team, fixture['team_h_difficulty'], 'H')
                team_to_gw26_details[fixture['team_a']] = (home_team, fixture['team_a_difficulty'], 'A')

            # Creating a list of player details including element type and opponent information with shortened opponent names
            user_players = [(player_id_to_details[pick['element']][0],  # Player name
                             player_id_to_details[pick['element']][1],  # Team ID
                             player_id_to_details[pick['element']][2],  # Element type
                             *team_to_gw26_details.get(player_id_to_details[pick['element']][1], ("Unknown Opponent", "Unknown Difficulty", "Unknown"))  # Opponent details
                            ) for pick in team_data['picks']]
            
            # Sorting players by element type
            user_players.sort(key=lambda x: x[2])

            # Updating opponent names to be shortened and stored in session
            team_to_gw26_details_short = {team_id: (opponent[:3].upper() if 'Manchester City' not in opponent else 'MCI', difficulty, home_or_away) for team_id, (opponent, difficulty, home_or_away) in team_to_gw26_details.items()}
            
            session['user_players'] = [(player[0], player[1], player[2], *team_to_gw26_details_short.get(player[1], ("Unknown Opponent", "Unknown Difficulty", "Unknown"))) for player in user_players]

        except Exception as e:
            print(f"An error occurred: {e}")

        return redirect(url_for('menu'))

    return render_template('login.html')

@app.route('/menu')
def menu():
    if 'fpl_id' in session:
        fpl_id = session['fpl_id']
        #retrive the 15 fpl players 
        user_players = session.get('user_players', ['Unknown'] * 15)

        player_name, whichFlag, team_name, bank, value = playerINFO2(fpl_id) #metrics in the API
        second_chance_rank = rank2Chance(fpl_id)
        currentHIGH, oldHIGH = BESTrank1(fpl_id)
        #chips
        chip_status = chipcheckers(fpl_id)
        #transfers
        free_transfers = freeTRANSFERcalc(fpl_id)

        league_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/")

        league_data = league_response.json()
        summary_overall_points = league_data.get('summary_overall_points', 0)
        overall_rank = league_data.get('summary_overall_rank', 'No rank')
        #rank and points, respectively. 
        overall_points = league_data.get('summary_event_points', 'No points here!')

        gameweeks = getHISTORY(fpl_id)
        finalRANKPERCENT = rankCHANGEpercent(gameweeks)
        rankUPorDOWN = "Up" if finalRANKPERCENT > 0 else "Down" if finalRANKPERCENT < 0 else "Does not change" #check if over 0
        bankdisp0 = f"{float(bank) / 10:.1f}" if bank != 'Nah' else bank #to display the amount the user has

        valuedisp0 = f"{float(value) / 10:.1f}" if value != 'ah' else value

        image_path = last10GWRANK(fpl_id)#fr pandas image

        # (Fetching rank differences) - then align with CSS classes
        ranklink = {
            "1m": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=20000",
            "100k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=2000",
            "10k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=200",
            "1k": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=20",
            "1": "https://fantasy.premierleague.com/api/leagues-classic/314/standings/?page_standings=1"
        }



        rank_differences = {}
        CSSrank_styles = {}
        for LAB1X, url in ranklink.items():
            rank_sort = int(LAB1X.replace('k', '000').replace('m', '000000'))
            total_for_rank = tL_sort(url, rank_sort)
            difference = total_for_rank - summary_overall_points #rank % diff

            rank_differences[LAB1X] = difference

            # CSS classes based on difference #progress bar
            if difference <= 0:
                CSSrank_styles[LAB1X] = 'rankstore'
            elif difference <= 15:
                CSSrank_styles[LAB1X] = 'rankstore90'
            elif difference <= 25:
                CSSrank_styles[LAB1X] = 'rankstore80'
            elif difference <= 40:
                CSSrank_styles[LAB1X] = 'rankstore70'
            elif difference <= 55:
                CSSrank_styles[LAB1X] = 'rankstore60'
            elif difference <= 75:
                CSSrank_styles[LAB1X] = 'rankstore50'
            elif difference <= 100:
                CSSrank_styles[LAB1X] = 'rankstore40'
            elif difference <= 130:
                CSSrank_styles[LAB1X] = 'rankstore30'
            elif difference <= 160:
                CSSrank_styles[LAB1X] = 'rankstore20'
            elif difference <= 190:
                CSSrank_styles[LAB1X] = 'rankstore10'
            else:
                CSSrank_styles[LAB1X] = 'rankstore0' #last rank (should be no line)
                

            #all returns summarised here. 
        return render_template('menu.html', player_name=player_name, whichFlag=whichFlag, team_name=team_name, second_chance_rank=second_chance_rank, fullOP=summary_overall_points, overall_rank=overall_rank, overall_points=overall_points, finalRANKPERCENT=finalRANKPERCENT, rankUPorDOWN=rankUPorDOWN, teamvalue=valuedisp0, inbank=bankdisp0, freetransfers=free_transfers, chip_status=chip_status, user_players=user_players, currentHIGH=currentHIGH, oldHIGH=oldHIGH, rank_differences=rank_differences, CSSrank_styles=CSSrank_styles, GimagePATH=image_path[len('static/'):])

    else: #quick error handling
        
        return render_template('menu.html', player_name='No player', GimagePATH=None)

@app.route('/KOTGW')
def KOTGW():
    if 'fpl_id' in session:
        fpl_id = session['fpl_id']
        #fav team 
        favteamDETAILS2 = favoteam(fpl_id)
        favteamDETAILS2['total_strength_overall'] = favteamDETAILS2.get('strength_overall_home', 0) + favteamDETAILS2.get('strength_overall_away', 0)
        favteamDETAILS2['total_strength_attack'] = favteamDETAILS2.get('strength_attack_home', 0) + favteamDETAILS2.get('strength_attack_away', 0)
        favteamDETAILS2['total_strength_defence'] = favteamDETAILS2.get('strength_defence_home', 0) + favteamDETAILS2.get('strength_defence_away', 0)

        # Use the team ID to get the team's goals
        goalsforthefav1 = hardcodegoals.get(favteamDETAILS2.get('favteam'), 0)

        players = predictedFAVTEAM(fpl_id)
        # Get the top three players for the favorite team
        if 'favteam' in favteamDETAILS2:
            top3Players = FullFAVTEAMplayers(favteamDETAILS2['favteam'])
        else:

            top3Players = ["No data found for the top three players!"] * 3  # Dh



        # Use the opponent team ID to get the opponent's goals
        goalsfortheopps1 = hardcodegoals.get(favteamDETAILS2.get('opponent_id', 0), 0)

        # Render the HTML template with team and opponent goals
        return render_template('KOTGW.html', favteamDETAILS2=favteamDETAILS2, players=players, top3Players=top3Players, goalsforthefav1=goalsforthefav1, goalsfortheopps1=goalsfortheopps1)
    else:
        return "You've not logged in!", 401


@app.route('/myai')
def myai():
    if 'fpl_id' in session:
        fpl_id = session['fpl_id']
        # Fetch player and game data
        # I have copied all this from menu. 
        player_name, whichFlag, team_name, bank, value = playerINFO2(fpl_id)
        second_chance_rank = rank2Chance(fpl_id)
        currentHIGH, oldHIGH = BESTrank1(fpl_id)
        chip_status = chipcheckers(fpl_id)
        free_transfers = freeTRANSFERcalc(fpl_id)
        league_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/")
        league_data = league_response.json()
        summary_overall_points = league_data.get('summary_overall_points', 0)
        overall_rank = league_data.get('summary_overall_rank', 'Not available')
        overall_points = league_data.get('summary_event_points', 'Not available')
        gameweeks = getHISTORY(fpl_id)
        finalRANKPERCENT = rankCHANGEpercent(gameweeks)
        rankUPorDOWN = "Up" if finalRANKPERCENT > 0 else "Down" if finalRANKPERCENT < 0 else "No change"
        bankdisp0 = f"{float(bank) / 10:.1f}" if bank != 'Not available' else bank
        valuedisp0 = f"{float(value) / 10:.1f}" if value != 'Not available' else value
        rank_differences, CSSrank_styles = diff_2RANK(summary_overall_points)
        predictedNumberFINAL = TOTALpredpoints(fpl_id)
        predictedNumberFINAL = round(predictedNumberFINAL, 2)
        difference = 117.44 - predictedNumberFINAL
        fpl_data = playerPicks1(fpl_id)
        fpl_data = playerPicks1(fpl_id)
        lowestplayers = lowestplayerFind(fpl_data, final_data)
        highest_player = best_playerINWHOLETEAM(fpl_data, final_data)
        
        player3AFFORD = affordedp(lowestplayers, final_data, bank)



        lowest_defender = lowestplayerFind(fpl_data, final_data)['DEF']
        lowest_fwd = lowestplayerFind(fpl_data, final_data)['FWD']
        lowest_mid = lowestplayerFind(fpl_data, final_data)['MID']


        # lowest defender
        defender_details = {

                'name': lowest_defender['name'],
                'predicted_points': lowest_defender['predicted_total_points'],
                'cost': lowest_defender['cost'],#
            }

        #lowest midfielder
        mid_details = {
                'name': lowest_mid['name'],
                'predicted_points': lowest_mid['predicted_total_points'],
                'cost': lowest_mid['cost']
            }

        #lowest forward
        if lowest_fwd is not None: #quick error test
            fwd_details = {
                'name': lowest_fwd['name'],
                'predicted_points': lowest_fwd['predicted_total_points'],
                    'cost': lowest_fwd['cost']
            }
        
        # for the rank prediction.
        data = DATAFETCH(fpl_id)
        if data:

            rank_data = EXTRACTDAT(data, fpl_id)
            if rank_data:
                df = pd.DataFrame(rank_data)
                df_processed = COMPUTEIT(df)


                # Simulate predictions and calculate percentages
                avg_prediction, best_prediction, worst_prediction, percentages = SIMPREDS(model, scaler, df_processed, n_simulations=100, BASEnoise=0.05, bias2fac=0.7, WORST10OUT=10)
            else:

                avg_prediction, best_prediction, worst_prediction, percentages = "X", "X", "X", {}


        else: #error handle
            return render_template('error.html', message="No data for this ID, you can try log in!")
        
        # final template. 
        return render_template('myai.html', player_name=player_name, whichFlag=whichFlag, team_name=team_name,
                               second_chance_rank=second_chance_rank, fullOP=summary_overall_points, overall_rank=overall_rank,
                               overall_points=overall_points, finalRANKPERCENT=finalRANKPERCENT, rankUPorDOWN=rankUPorDOWN,
                               teamvalue=valuedisp0, inbank=bankdisp0, freetransfers=free_transfers, chip_status=chip_status,
                               #print team below
                               user_players=session.get('user_players', ['Unknown'] * 15), currentHIGH=currentHIGH,
                               oldHIGH=oldHIGH, rank_differences=rank_differences, CSSrank_styles=CSSrank_styles,
                               avg_prediction=avg_prediction, best_prediction=best_prediction, worst_prediction=worst_prediction,
                               percentages=percentages, predictedNumberFINAL=predictedNumberFINAL, difference=difference, defender=defender_details, mid=mid_details, fwd=fwd_details, player3AFFORD=player3AFFORD, bank=bank, highest_player=highest_player)
  
  
    else: #if not, go back to login.
        return redirect(url_for('login'))


#stats section here
@app.route('/stats')

def stats():
    
    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    data = response.json()

    players = data['elements']

    #sort 1-5
    def SORTbyPLAYfinal(sort_key, REQCOUNTS):
        groupPlays = {etype: [] for etype in range(1, 5)}

        for player in players:
            etype = player['element_type'] #base on element type again
                    #appends the player to the list of best players. 
            groupPlays[etype].append(player)

        for etype in groupPlays:
            groupPlays[etype].sort(key=lambda x: float(x[sort_key]), reverse=True) #back to front!

        selected3 = []
        for etype, count in REQCOUNTS.items():
            selected3.extend(groupPlays[etype][:count])

            #returns

        return [{
            'name': player['web_name'],
            'team_id': player['team'],
            'selected_by_percent': player['selected_by_percent'],
            'transfers_in_event': player['transfers_in_event'],
            'transfers_out_event': player['transfers_out_event'],
            'net_transfers': player['transfers_in_event'] - player['transfers_out_event'],
            'form': player['form'],
            'goals_scored': player['goals_scored'],
            'expected_goals': player['expected_goals'],
            'assists': player['assists'],
            'expected_goal_involvements': player['expected_goal_involvements'],
            'ict_index': player['ict_index'],
            'total_points': player['total_points'],
            'value_season': player['value_season'],
            'clean_sheets': player['clean_sheets']
        } for player in selected3]
    


    def under10points():
        groupPlays = {etype: [] for etype in range(1, 5)}
        for player in players:
            etype = player['element_type']
            if float(player['selected_by_percent']) < 10:
                groupPlays[etype].append(player)

        for etype in groupPlays:
            groupPlays[etype].sort(key=lambda x: float(x['total_points']), reverse=True)

        REQCOUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
        REQCOUNTS2 = {1: 2, 2: 13, 3: 0, 4: 0}
        selected3 = []
        for etype, count in REQCOUNTS.items():
            selected3.extend(groupPlays[etype][:count])

        return [{
            'name': player['web_name'],
            'team_id': player['team'],
            #continuted metrics

            'selected_by_percent': player['selected_by_percent'],
            'transfers_in_event': player['transfers_in_event'],
            'transfers_out_event': player['transfers_out_event'],
            'net_transfers': player['transfers_in_event'] - player['transfers_out_event'],
            'form': player['form'],
            'goals_scored': player['goals_scored'],
            'expected_goals': player['expected_goals'],
            'assists': player['assists'],
            'expected_goal_involvements': player['expected_goal_involvements'],
            'ict_index': player['ict_index'],
            'total_points': player['total_points'],
            'value_season': player['value_season'],
            'clean_sheets': player['clean_sheets']
        } for player in selected3]

    REQCOUNTS = {1: 2, 2: 5, 3: 5, 4: 3} #the counts for each position
    REQCOUNTS2 = {1: 2, 2: 13, 3: 0, 4: 0} 



    #all metrics on the API
    players_by_selected_percent = SORTbyPLAYfinal('selected_by_percent', REQCOUNTS)
    players_by_transfers_in = SORTbyPLAYfinal('transfers_in_event', REQCOUNTS)
    playersbyform = SORTbyPLAYfinal('form', REQCOUNTS)
    playergoals = SORTbyPLAYfinal('goals_scored', REQCOUNTS)
    expectedgoals = SORTbyPLAYfinal('expected_goals', REQCOUNTS)
    assists = SORTbyPLAYfinal('assists', REQCOUNTS)
    xgi = SORTbyPLAYfinal('expected_goal_involvements', REQCOUNTS)
    ictindex = SORTbyPLAYfinal('ict_index', REQCOUNTS)
    TOTALpoints = SORTbyPLAYfinal('total_points', REQCOUNTS)
    valueseason = SORTbyPLAYfinal('value_season', REQCOUNTS)
    clean_sheets = SORTbyPLAYfinal('clean_sheets', REQCOUNTS2)

    under10per = under10points()

    teams_data = {
        'highest_owned_players': players_by_selected_percent,
        'transfers_in_players': players_by_transfers_in,
        'playersbyform': playersbyform,
        'playergoals': playergoals,
        'expectedgoals': expectedgoals,
        'assists': assists,
        'xgi': xgi,
        'ictindex': ictindex,
        'TOTALpoints': TOTALpoints,
        'valueseason': valueseason,
        'clean_sheets': clean_sheets,
        'under10per': under10per
        #finish
    }

    # Finds the players with the highest points (for each metric)

    TOPHIGH1 = max(TOTALpoints, key=lambda x: x['total_points'])
    TOPHIGH2 = max(players_by_selected_percent, key=lambda x: x['selected_by_percent'])
    TOPHIGH3 = max(players_by_transfers_in, key=lambda x: x['transfers_in_event'])
    TOPHIGH4 = max(playersbyform, key=lambda x: x['form'])
    TOPHIGH5 = max(playergoals, key=lambda x: x['goals_scored'])
    TOPHIGH6 = max(expectedgoals, key=lambda x: x['expected_goals'])
    TOPHIGH7 = max(assists, key=lambda x: x['assists'])
    TOPHIGH8 = max(xgi, key=lambda x: x['expected_goal_involvements'])
    TOPHIGH9 = max(ictindex, key=lambda x: x['ict_index'])
    TOPHIGH10 = max(valueseason, key=lambda x: x['value_season'])
    TOPHIGH11 = max(under10per, key=lambda x: x['total_points'])
    TOPHIGH12 = max(clean_sheets, key=lambda x: x['clean_sheets'])


    
    playersLeft = [player for player in TOTALpoints if player['name'] != TOPHIGH1['name']]
    playersLeft2 = [player2 for player2 in players_by_selected_percent if player2['name'] != TOPHIGH2['name']]
    playersLeft3 = [player3 for player3 in players_by_transfers_in if player3['name'] != TOPHIGH3['name']]
    playersLeft4 = [player4 for player4 in playersbyform if player4['name'] != TOPHIGH4['name']]
    playersLeft5 = [player5 for player5 in playergoals if player5['name'] != TOPHIGH5['name']]
    playersLeft6 = [player6 for player6 in expectedgoals if player6['name'] != TOPHIGH6['name']]
    playersLeft7 = [player7 for player7 in assists if player7['name'] != TOPHIGH7['name']]
    playersLeft8 = [player for player in xgi if player['name'] != TOPHIGH8['name']]
    playersLeft9 = [player for player in ictindex if player['name'] != TOPHIGH9['name']]
    playersLeft10 = [player for player in valueseason if player['name'] != TOPHIGH10['name']]
    playersLeft11 = [player for player in under10per if player['name'] != TOPHIGH11['name']]
    playersLeft12 = [player for player in clean_sheets if player['name'] != TOPHIGH12['name']]

    # Find the player with the second-highest total points among the remaining players
    SECHIGH1 = max(playersLeft, key=lambda x: x['total_points'])
    SECHIGH2 = max(playersLeft2, key=lambda x: x['selected_by_percent'])
    SECHIGH3 = max(playersLeft3, key=lambda x: x['transfers_in_event'])
    SECHIGH4 = max(playersLeft4, key=lambda x: x['form'])
    SECHIGH5 = max(playersLeft5, key=lambda x: x['goals_scored'])
    SECHIGH6 = max(playersLeft6, key=lambda x: x['expected_goals'])
    SECHIGH7 = max(playersLeft7, key=lambda x: x['assists'])
    SECHIGH8 = max(playersLeft8, key=lambda x: x['expected_goal_involvements'])
    SECHIGH9 = max(playersLeft9, key=lambda x: x['ict_index'])
    SECHIGH10 = max(playersLeft10, key=lambda x: x['value_season'])
    SECHIGH11 = max(playersLeft11, key=lambda x: x['total_points'])
    SECHIGH12 = max(playersLeft12, key=lambda x: x['clean_sheets'])

    # Exclude the player with the second-highest total points
    playersLeft = [player for player in playersLeft if player['name'] != SECHIGH1['name']]
    playersLeft2 = [player2 for player2 in playersLeft2 if player2['name'] != SECHIGH2['name']]
    playersLeft3 = [player3 for player3 in playersLeft3 if player3['name'] != SECHIGH3['name']]
    playersLeft4 = [player4 for player4 in playersLeft4 if player4['name'] != SECHIGH4['name']]
    playersLeft5 = [player5 for player5 in playersLeft5 if player5['name'] != SECHIGH5['name']]
    playersLeft6 = [player6 for player6 in playersLeft6 if player6['name'] != SECHIGH6['name']]
    playersLeft7 = [player7 for player7 in playersLeft7 if player7['name'] != SECHIGH7['name']]
    playersLeft8 = [player for player in playersLeft8 if player['name'] != SECHIGH8['name']]
    playersLeft9 = [player for player in playersLeft9 if player['name'] != SECHIGH9['name']]
    playersLeft10 = [player for player in playersLeft10 if player['name'] != SECHIGH10['name']]
    playersLeft11 = [player for player in playersLeft11 if player['name'] != SECHIGH11['name']]
    playersLeft12 = [player for player in playersLeft12 if player['name'] != SECHIGH12['name']]

    #still iterates (third time)
    THIRDHIGH1 = max(playersLeft, key=lambda x: x['total_points'])
    THIRDHIGH2 = max(playersLeft2, key=lambda x: x['selected_by_percent'])
    THIRDHIGH3 = max(playersLeft3, key=lambda x: x['transfers_in_event'])
    THIRDHIGH4 = max(playersLeft4, key=lambda x: x['form'])
    THIRDHIGH5 = max(playersLeft5, key=lambda x: x['goals_scored'])
    THIRDHIGH6 = max(playersLeft6, key=lambda x: x['expected_goals'])
    THIRDHIGH7 = max(playersLeft7, key=lambda x: x['assists'])
    THIRDHIGH8 = max(playersLeft8, key=lambda x: x['expected_goal_involvements'])
    THIRDHIGH9 = max(playersLeft9, key=lambda x: x['ict_index'])
    THIRDHIGH10 = max(playersLeft10, key=lambda x: x['value_season'])
    THIRDHIGH11 = max(playersLeft11, key=lambda x: x['total_points'])
    THIRDHIGH12 = max(playersLeft12, key=lambda x: x['clean_sheets'])

    # Pass both highest and second-highest total points players to the template
    return render_template('stats.html', teams_data=teams_data, TOPHIGH1=TOPHIGH1, SECHIGH1=SECHIGH1, THIRDHIGH1=THIRDHIGH1, TOPHIGH2=TOPHIGH2, SECHIGH2=SECHIGH2, THIRDHIGH2=THIRDHIGH2, TOPHIGH3=TOPHIGH3, SECHIGH3=SECHIGH3, THIRDHIGH3=THIRDHIGH3, TOPHIGH4=TOPHIGH4, SECHIGH4=SECHIGH4, THIRDHIGH4=THIRDHIGH4, TOPHIGH5=TOPHIGH5, SECHIGH5=SECHIGH5, THIRDHIGH5=THIRDHIGH5, TOPHIGH6=TOPHIGH6, SECHIGH6=SECHIGH6, THIRDHIGH6=THIRDHIGH6, TOPHIGH7=TOPHIGH7, SECHIGH7=SECHIGH7, THIRDHIGH7=THIRDHIGH7, TOPHIGH8=TOPHIGH8, SECHIGH8=SECHIGH8, THIRDHIGH8=THIRDHIGH8, TOPHIGH9=TOPHIGH9, SECHIGH9=SECHIGH9, THIRDHIGH9=THIRDHIGH9, TOPHIGH10=TOPHIGH10, SECHIGH10=SECHIGH10, THIRDHIGH10=THIRDHIGH10, TOPHIGH11=TOPHIGH11, SECHIGH11=SECHIGH11, THIRDHIGH11=THIRDHIGH11, TOPHIGH12=TOPHIGH12, SECHIGH12=SECHIGH12, THIRDHIGH12=THIRDHIGH12)


@app.route('/experts')
def experts():

    # the expert fpl IDs
    fpl_ids = [4449, 27371, 139820, 407607]

    allusersX = {}
    overall_ranks = {} 

    for fpl_id in fpl_ids:
        try:
            # Fetches TEAM DATA
            team_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/38/picks/")
            team_data = team_response.json()


            # Bootstrap initialise ing
            response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
            data = response.json()

            league_response = requests.get(f"https://fantasy.premierleague.com/api/entry/{fpl_id}/")
            league_data = league_response.json()
            overall_ranks[fpl_id] = league_data.get('summary_overall_rank', 'N/a')

            # Player & team aligning
            playerIDtoDETAILS = {player['id']: (player['web_name'], player['team'], player['element_type']) for player in data['elements']}
            #to name
            teamID2name = {team['id']: team['name'] for team in data['teams']}

            # getthefixtures
            FINALFIXRESPONSE = requests.get("https://fantasy.premierleague.com/api/fixtures/")
            fixturesfinal = [fixture for fixture in FINALFIXRESPONSE.json() if fixture['event'] == 38] # for 32. 

            team2gameweek = {}
            for fixture in fixturesfinal:
                home_team = teamID2name[fixture['team_h']]
                away_team = teamID2name[fixture['team_a']]
                calcers = len(team2gameweek) * 0

                # Determine if home or away.
                team2gameweek[fixture['team_h']] = (away_team, fixture['team_h_difficulty'], 'H')
                team2gameweek[fixture['team_a']] = (home_team, fixture['team_a_difficulty'], 'A')  # Same structure


            # final user players list

            user_players = [(playerIDtoDETAILS[pick['element']][0],  # Player Name
                             playerIDtoDETAILS[pick['element']][1],  # Team ID
                             playerIDtoDETAILS[pick['element']][2],  # Element type
                             *team2gameweek.get(playerIDtoDETAILS[pick['element']][1], ("X", "X", "X")) 
                            ) for pick in team_data['picks']]
            
            #final list sorted
            user_players.sort(key=lambda x: x[2])

            # Shorten opponent names, special case for Manchester City
            team2gameweekSHORT = {team_id: (opponent[:3].upper() if 'Manchester City' not in opponent else 'MCI', difficulty, home_or_away) for team_id, (opponent, difficulty, home_or_away) in team2gameweek.items()}
            
            # Store processed data for each fpl_id
            allusersX[fpl_id] = [(player[0], player[1], player[2], *team2gameweekSHORT.get(player[1], ("X", "X", "?"))) for player in user_players]



        except Exception as e: #final expection. 
            print(f"An error occurred for fpl_id {e}")
            allusersX[fpl_id] = ['Unknown'] * 15

    # Pass dictionary to template
    return render_template('experts.html', teams_data=allusersX, overall_ranks=overall_ranks)

if __name__ == '__main__':
    app.run(debug=True)
