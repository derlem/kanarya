from django import forms


class QueryForm(forms.Form):
	sentence = forms.CharField(max_length=500)
	is_error_found = forms.BooleanField(required=False)
	tagged_string = forms.TextInput()


class QueryFeedbackForm(forms.Form):
	report = forms.CharField(max_length=1000)