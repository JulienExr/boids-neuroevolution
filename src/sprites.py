import pygame


def create_bird_sprite() -> pygame.Surface:
    sprite = pygame.Surface((26, 18), pygame.SRCALPHA)
    body = [(24, 9), (2, 2), (8, 9), (2, 16)]
    pygame.draw.polygon(sprite, (233, 241, 255), body)
    pygame.draw.polygon(sprite, (88, 167, 255), body, 2)
    pygame.draw.circle(sprite, (255, 255, 255), (18, 8), 2)
    return sprite.convert_alpha()


def create_predator_sprite() -> pygame.Surface:
    sprite = pygame.Surface((42, 30), pygame.SRCALPHA)
    body = [(40, 15), (3, 3), (13, 15), (3, 27)]
    pygame.draw.polygon(sprite, (255, 103, 82), body)
    pygame.draw.polygon(sprite, (82, 19, 28), body, 3)
    pygame.draw.circle(sprite, (255, 242, 166), (29, 13), 3)
    return sprite.convert_alpha()
