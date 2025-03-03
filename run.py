from LED import *
from player import player
from level.level import level
from src.gui import hud, end_screen
from src.lore import display_lore
from src.game import game

def update():
    game.update()

    if get_key_pressed('r'):
        game.reset()
        level.generate()

    if player.hp <= 0: # not game.asteroids or
        return game.score

    draw_entities()
    update_entities()
    return None

def draw_entities():
    for s in level.stars:
        s.draw_self()

    for p in level.planets:
        p.draw_self()
        
    for a in game.asteroids:
        a.draw_self()
        
    for p in game.particles:
        p.draw_self()

    for b in game.bullets:
        b.draw_self()
    
    player.draw_self()

def update_entities():
    for b in list(game.bullets):
        b.update()

    player.update()

def main():
    option = "Play"

    # if level.level_number == 0:
    #     display_lore()

    while True:
        if option == "Play":
            game.reset()
            player.hp = 100
            level.generate()

            while True:
                refresh()
                
                score = update()

                if score is not None:
                    break
                
                hud()
                draw()
            
            option = end_screen(score)
        elif option == "Quit":
            return
        else:
            raise ValueError("Invalid option given in game")

if __name__ == "__main__":
    main()
