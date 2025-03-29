# Markdown Cheat Sheet

Thanks for visiting [The Markdown Guide](https://www.markdownguide.org)!

This Markdown cheat sheet provides a quick overview of all the Markdown syntax elements. It can’t cover every edge case, so if you need more information about any of these elements, refer to the reference guides for [basic syntax](https://www.markdownguide.org/basic-syntax) and [extended syntax](https://www.markdownguide.org/extended-syntax).

## Basic Syntax

These are the elements outlined in John Gruber’s original design document. All Markdown applications support these elements.

### Heading

# H1
## H2
### H3

### Bold

**bold text**

### Italic

*italicized text*

### Blockquote

> blockquote

### Ordered List

1. First item
2. Second item
3. Third item

### Unordered List

- First item
- Second item
- Third item

### Code

`code`

### Horizontal Rule

---

### Link

[Markdown Guide](https://www.markdownguide.org)

### Image

![alt text](https://www.markdownguide.org/assets/images/tux.png)

## Extended Syntax

These elements extend the basic syntax by adding additional features. Not all Markdown applications support these elements.

### Table

| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |

### Fenced Code Block

```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```

### Footnote

Here's a sentence with a footnote. [^1]

[^1]: This is the footnote.

### Heading ID

### My Great Heading {#custom-id}

### Definition List

term
: definition

### Strikethrough

~~The world is flat.~~

### Task List

- [x] Write the press release
- [ ] Update the website
- [ ] Contact the media

### Go to next line

This will split the sentence \
into two lines

In the above format, ensure there is no whitespace between the backslash and the next word (in this case 'into')

# Button size

Wrap the button in a container that uses the font-size utility classes to scale buttons:

<div class="code-example" markdown="1">
<span class="fs-6">
[Big ass button](http://example.com/){: .btn }
</span>

<span class="fs-3">
[Tiny ass button](http://example.com/){: .btn }
</span>
</div>
```markdown
<span class="fs-8">
[Link button](http://example.com/){: .btn }
</span>

<span class="fs-3">
[Tiny ass button](http://example.com/){: .btn }
</span>

### Add Image with Caption (modelled as a table)

| ![](data/rendering_pipeline.png) |
|:--:|
| *Accelerating 3D Deep Learning with PyTorch3D. Ravi et. al.* |

### Add MathJax support

# MathJax v3 Configuration

In `_includes/head_custom.html` add, for example:

{% raw %}
```html
{% case page.math %}

  {% when "mathjax3" %}

    <script>
      MathJax = { 
        tex: { 
          tags: 'ams',
          packages: {'[+]': ['textmacros']},
        },
        loader: {
          load: ['[tex]/textmacros']
        }
      };
    </script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>

{% endcase %}
```
{% endraw %}

See also [further MathJax v3 configuration options](http://docs.mathjax.org/en/latest/web/configuration.html).

In the front matter of pages using MathJax v3 (or as a global front-matter default) add:

```yaml
layout: default
title: Homework 4
nav_order: 1
description: Cats Generator Playground
permalink: /
math: mathjax3
```

(The suggested field name `math` and the key `mathjax3` can be replaced.)


After that, simply wrap the math symbols with two double-dollar sign ```$$```
```
$$E=mc^2$$
```

For further examples visit: https://github.com/pdmosses/just-the-docs-tests-old/blob/master/docs/math/mathjax3/tests.md

