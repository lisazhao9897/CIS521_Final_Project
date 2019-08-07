# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# get text input from user 
from tkinter import *
user_entry = '' 

def add_text():
    label1 = Label(root, text = "Alright :)")
    label1.pack()
    global user_entry
    user_entry = text_box.get()
    root.after(2000, lambda : root.destroy())

root = Tk()
root.title("R2D2 Chat Window")
root.geometry("300x150")

text_label = Label(root, text="What do you want to tell me...")
text_label.pack()

text_box = Entry(root, bd=1)
text_box.pack()

enter_button = Button(root, text="OK", command=add_text)
enter_button.pack()

root.mainloop()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# connect to R2D2 
from client import DroidClient
droid = DroidClient()
droid.connect_to_droid('Q5-F43E')
droid.animate(5)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import string 
import random 
import numpy as np
import copy

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# NLP functions  
def tokenize(text):
    out = [] 
    tokens = text.split() 

    for token in tokens: 
        t = ''
        for char in token: 
            # char in a punc 
            if char in string.punctuation: 
                # return prev char in this token if not empty 
                if t != '': out.append(t)
                # return this char 
                out.append(char)
                # reset t = ''
                t = '' 
            # char NOT a punc  
            else: 
                t += char 
        
            # in the end, return this token 
        if t != '': out.append(t)
            
    return out 

def ngrams(n, tokens):
    out = [] 
    new_text = ['<START>'] * (n-1) 
    new_text.extend(tokens) 
    new_text.extend(['<END>'])

    for idx in range(n-1,len(new_text)):
        context = tuple(new_text[idx-n+1:idx])
        token = new_text[idx]
        out.append((context,token))
    return out 

class NgramModel(object):
    def __init__(self, n):
        self.n = n
        # (context, token): count 
        self.ct_table = {}
        
        # context: count 
        self.c_table = {}

    def update(self, sentence):
        tokens = tokenize(sentence)
        ngram_result = ngrams(self.n, tokens)
        
        for (context, token) in ngram_result: 
            # this context WAS seen 
            if context in self.c_table: 
                self.c_table[context] += 1 
                # this (context, token) WAS seen 
                if (context, token) in self.ct_table: 
                    self.ct_table[(context, token)] += 1 
                # context SEEN, (context, token) NOT 
                else: 
                    self.ct_table[(context, token)] = 1 
            
            # this context NOT seen 
            else: 
                self.c_table[context] = 1 
                self.ct_table[(context, token)] = 1 

    def smoothed_prob(self, sentence): 
        pr = 1 
        
        tokens = ['<START>'] * (self.n-1)
        tokens.extend(tokenize(sentence))
        tokens.extend(['<END>'])
        
        context = tuple(['<START>'] * (self.n-1)) 
        for token in tokens[self.n-1:]:
            if context not in self.c_table: 
                # penalize 
                this_prob = 0.01 
            else:  
                bot = self.c_table[context] 
                if (context, token) not in self.ct_table: 
                    top = 0 
                else: 
                    top = self.ct_table[(context, token)]
                 
                num_tokens = 0 
                for (c, t) in self.ct_table.keys():
                    if c == context: 
                        num_tokens += 1 
                    
                this_prob = (top + 1) / (bot + num_tokens) 
            
            pr = pr * this_prob 
            # update context 
            context = context[1:] + (token, )
        return pr
    
    def get_table(self): 
        return self.ct_table

def get_target(n, food_text, friend_text, sentence):
    sentence = sentence.lower() 
    food_ngram_mdl = NgramModel(n)
    food_ngram_mdl.update(food_text.lower())
    
    friend_ngram_mdl = NgramModel(n)
    friend_ngram_mdl.update(friend_text.lower())
    
    food_pr = food_ngram_mdl.smoothed_prob(sentence)
    friend_pr = friend_ngram_mdl.smoothed_prob(sentence)
    
    if food_pr > friend_pr: 
        return 'cake'
    return 'friend'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A* functions 
def dist(A,B): 
    # A: (a1, a2), B: (b1, b2)
    a1, a2 = A[0], A[1]
    b1, b2 = B[0], B[1]
    out = np.sqrt((a1-b1) ** 2 + (a2-b2) ** 2)
    return out 

def allowed_actions(A, scene):
    h, w = A[0], A[1]
    s_row, s_col = len(scene), len(scene[0])
    out = [] 
    # left, up, right, down 
    possible_h = [h, h-1, h, h+1] 
    possible_w = [w-1, w, w+1, w]
  
    for i in range(len(possible_h)): 
        if possible_h[i] >= 0 and possible_h[i] < s_row and \
        possible_w[i] >= 0 and possible_w[i] < s_col and \
        scene[possible_h[i]][possible_w[i]] != 'W': 
            out.append((possible_h[i], possible_w[i]))
    return out 

def find_goal(scene, goal):
    if goal == 'cake': 
        char = 'C'
    else: 
        char = 'R'
    h, w = len(scene), len(scene[0])
    for i in range(h): 
        for j in range(w): 
            if grid[i][j] == char: 
                return ((i,j))

def find_path(start, goal, scene):
    print('goal: ', goal)
    # if the start point or goal point lies on an obstacle, 
    if scene[start[0]][start[1]] == 'W' or scene[goal[0]][goal[1]] == 'W': 
        return None 
 
    # priority queue --> node: current f 
    frontiers = {start: 0 + dist(start, goal)}
    # node: {parent, current min g to get to this node}
    explored = {start: (None, 0)}
   
    while len(frontiers) > 0:
        # find the node with min f on frontiers, q, pop q off frontiers 
        sorted_frontiers = sorted(frontiers.items(), key=lambda kv: kv[1])
        now_state, now_state_f = sorted_frontiers[0][0], sorted_frontiers[0][1]
        del frontiers[now_state]
    
        # goal test 
        if now_state == goal: 
            out = [goal] 
            prev = goal
        
            while explored[prev][0] != None:
                prev = explored[prev][0]
                out.insert(0,prev)
            return out 
    
        # now is not goal 
        else: 
            # add now_state children to frontier 
            for child in allowed_actions(now_state, scene):
                # child f = parent g + dist(parent, child) + child h 
                this_child_g = explored[now_state][1] + dist(now_state, child) 
                this_child_f = this_child_g + dist(child, goal)
                # in explored 
                if child in explored and explored[child][1] > this_child_g:
                    explored[child] = (now_state, this_child_g)
                    # also in frontiers now 
                    if child in frontiers: 
                        frontiers[child] = this_child_f 
                # yet explored --> add to explored & frontiers     
                elif child not in explored:  
                    explored[child] = (now_state, this_child_g)
                    frontiers[child] = this_child_f
                    
    # if no optimal solutions exist 
    return None 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def path_to_action(trail, speed, roll_time): 
    print('trail: ', trail)
    for i in range(1, len(trail)):
        now, before = trail[i], trail[i-1]
        change_h = now[0] - before[0]
        change_w = now[1] - before[1]
        
        print('change_h: ', change_h)
        print('change_w: ', change_w)

        if change_h == 1 and change_w == 0: 
            droid.roll(dir_speed['down'], rotate_deg['down'], roll_time) 

        elif change_h == -1 and change_w == 0:   
            droid.roll(dir_speed['up'], rotate_deg['up'], roll_time) 
        
        elif change_h == 0 and change_w == 1: 
            droid.roll(dir_speed['right'], rotate_deg['right'], roll_time) 

        else: 
            droid.roll(dir_speed['left'], rotate_deg['left'], roll_time) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PARAMETERS AND ENVIRONMENT 
start = (0,0)

speed = 0.2 
roll_time = 1 

# grid = [[' ',' ',' ',' ', ' '],
#         [' ','W',' ','W', 'W'],
#         [' ','W',' ',' ', 'W'],
#         [' ','W','W',' ', ' '],
#         [' ','R',' ',' ', 'C']]

grid = [[' ',' ',' ',' ', ' '],
        [' ','W','W','W', 'W'],
        [' ',' ',' ',' ', 'W'],
        [' ','W','W',' ', ' '],
        [' ','R',' ',' ', 'C']]

food_text = 'I like cake. I like chocolate. I like ice cream. Mostly I like cake. \
I like to eat. I eat when I am really hungry. Food smells very good. Ice cream is taste.'

friend_text = 'I am bored. I play with my friends. My friends are John, Mary, and Lily. \
If my friend is at home, I play with him.'

food_sentences = ['I am really hungry', 
                  'I eat ice cream',
                  'I want to eat', 
                  'Cake or ice cream']

friend_sentences = ['My friend is waiting', 
                    'I am so bored', 
                    'John is at home', 
                    'I play with them'] 

rotate_deg = {
    'up': 0, 
    'right': 90, 
    'down': 180, 
    'left': 270
}

dir_speed = {
    'up': 0.09, 
    'right': 0.14, 
    'down': 0.095, 
    'left': 0.14
}

# for s in food_sentences: 
#     print(get_target(3, food_text, friend_text, s)) 
    
# for s in friend_sentences: 
#     print(get_target(3, food_text, friend_text, s)) 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
target = get_target(3, food_text, friend_text, user_entry)

goal = find_goal(grid, target)

trail = find_path(start, goal, grid)

path_to_action(trail, speed, roll_time)
