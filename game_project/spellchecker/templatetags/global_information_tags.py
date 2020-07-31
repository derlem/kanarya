from django import template

register = template.Library()

from game_project.settings import DOMAIN_NAME

@register.simple_tag
def domain_name():
    return DOMAIN_NAME