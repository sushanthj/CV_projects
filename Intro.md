---
layout: page
title: Building this Page
permalink: /intro/
nav_order: 2
---

<!-- # Computer Vision References

[Main Vision Reference](https://szeliski.org/Book/){: .btn .fs-3 .mb-4 .mb-md-0 }

[Reference Book 1](https://drive.google.com/file/d/1jqEB739EfifhSyiCK6vdbPIz7gX9Ywmr/view?usp=sharing){: .btn .fs-3 .mb-4 .mb-md-0 }
[Reference Book 2](https://drive.google.com/file/d/1Kn6dilDeR_7leIctuVa87-czuqBoxJh-/view?usp=sharing){: .btn .fs-3 .mb-4 .mb-md-0 }
 -->

# Bulding the Webpage

For Jekyll reference see [just_the_docs](https://pmarsceill.github.io/just-the-docs/)


The following pages are built in order to understand Computer Vision and Machine Learning

To deploy on heroku follow the steps in the link below (and use the gem files, rake files and proc files in this repo for reference)

The following files will need to be copied from this repo:
- config.ru
- Rakefile
- Procfile
- static.json
- config.yaml (modify this file as per requirement)
- Gemfile

And only if necessary:
- Gemfile.lock
- remove _sites from .gitignore

Run bundle once to intialize
Run bundle exec jekyll serve
Go to the specified webpage by the above command

After copying these files (or their necessary contents), install heroku cli and do heroku login:
```bash
curl https://cli-assets.heroku.com/install.sh | sh
heroku login
```

Then directly start with heroku create as per the below link and the other steps necessary (git push heroku master)

[Deploy jekyll on heroku](https://blog.heroku.com/jekyll-on-heroku)

Finally, go to heroku page -> settings -> change the name of the app and find the url

# To better your experience of writing in code

Download the following extensions in vscode:
1. Markdown All in one
2. code runner (see youtube video on how to setup vscode for C++)


# Shortcuts in general pour toi
- Once Markdown all in one is installed, you can do **ctrl+shift+v** to see preview of markdown immediately
- To run any C++ file it's just **ctrl+shift+n**
- If you want to bold any specific text in markdown just select the text by holding down **ctrl+shift** and using arrow keys to select the required text. Then once text is selected just do **ctrl+b** to **bolden** and **ctrl+i** to ***italicize***
  - click on tab after using **-** for normal bullet pointing to get sub-points
- To get numbered list continuously, in-between two headings 1. and 2. all content should be indented with 4 spaces in the markdown script

- To shift between windows in ubuntu, just do **windows_key+shift+right/left_arrow**
- To minimize or unmaximize any window in **hold down alt and press space**, then choose to minimize
- To then maximize or move window to right half/left half of screen, **windows_key+shift+right/left_arrow**
