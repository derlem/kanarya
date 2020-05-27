from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Profile

from django.utils.safestring import mark_safe

class UserRegisterForm(UserCreationForm):
	email = forms.EmailField()
	tos = forms.BooleanField(label=mark_safe('<a href="/onamformu" target="_blank">Onam Formu</a>nu okudum ve onaylıyorum.'))

	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2', 'tos']



class UserUpdateForm(forms.ModelForm):
	email = forms.EmailField()

	class Meta:
		model = User
		fields = ['username', 'email']


class ProfileUpdateForm(forms.ModelForm):
	class Meta:
		model = Profile
		fields = ['image']

