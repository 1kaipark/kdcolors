import json, re 
from typing import Literal

# this is scuffed asf, but dealing with relative paths was annoying so here's the palettes as strings lmfao

gruvbox_palette = """
:root {
  --bg_h: #1d2021;
  --bg:   #282828;
  --bg_s: #32302f;
  --bg1:  #3c3836;
  --bg2:  #504945;
  --bg3:  #665c54;
  --bg4:  #7c6f64;

  --fg:  #fbf1c7;
  --fg1: #ebdbb2;
  --fg2: #d5c4a1;
  --fg3: #bdae93;
  --fg4: #a89984;

  --red:    #fb4934;
  --green:  #b8bb26;
  --yellow: #fabd2f;
  --blue:   #83a598;
  --purple: #d3869b;
  --aqua:   #8ec07c;
  --gray:   #928374;
  --orange: #fe8019;

  --red-dim:    #cc2412;
  --green-dim:  #98971a;
  --yellow-dim: #d79921;
  --blue-dim:   #458588;
  --purple-dim: #b16286;
  --aqua-dim:   #689d6a;
  --gray-dim:   #a89984;
  --orange-dim: #d65d0e;
}
"""

nord_css = """
/*
 * Copyright (c) 2016-present Sven Greb <development@svengreb.de>
 * This source code is licensed under the MIT license found in the license file.
 */

/*
 * References:
 *   1. https://www.w3.org/TR/css-variables
 *   2. https://www.w3.org/TR/selectors/#root-pseudo
 *   3. https://drafts.csswg.org/css-variables
 *   4. https://developer.mozilla.org/en-US/docs/Web/CSS/Using_CSS_variables
 *   5. https://warpspire.com/kss
 *   6. https://github.com/kss-node/kss-node
 */

/*
An arctic, north-bluish color palette.
Created for the clean- and minimal flat design pattern to achieve a optimal focus and readability for code syntax
highlighting and UI.
It consists of a total of sixteen, carefully selected, dimmed pastel colors for a eye-comfortable, but yet colorful
ambiance.

Styleguide Nord
*/

:root {
  /*
  Base component color of "Polar Night".

  Used for texts, backgrounds, carets and structuring characters like curly- and square brackets.

  Markup:
  <div style="background-color:#2e3440; width=60; height=60"></div>

  Styleguide Nord - Polar Night
  */
  --nord0: #2e3440;

  /*
  Lighter shade color of the base component color.

  Used as a lighter background color for UI elements like status bars.

  Markup:
  <div style="background-color:#3b4252; width=60; height=60"></div>

  Styleguide Nord - Polar Night
  */
  --nord1: #3b4252;

  /*
  Lighter shade color of the base component color.

  Used as line highlighting in the editor.
  In the UI scope it may be used as selection- and highlight color.

  Markup:
  <div style="background-color:#434c5e; width=60; height=60"></div>

  Styleguide Nord - Polar Night
  */
  --nord2: #434c5e;

  /*
  Lighter shade color of the base component color.

  Used for comments, invisibles, indent- and wrap guide marker.
  In the UI scope used as pseudoclass color for disabled elements.

  Markup:
  <div style="background-color:#4c566a; width=60; height=60"></div>

  Styleguide Nord - Polar Night
  */
  --nord3: #4c566a;

  /*
  Base component color of "Snow Storm".

  Main color for text, variables, constants and attributes.
  In the UI scope used as semi-light background depending on the theme shading design.

  Markup:
  <div style="background-color:#d8dee9; width=60; height=60"></div>

  Styleguide Nord - Snow Storm
  */
  --nord4: #d8dee9;

  /*
  Lighter shade color of the base component color.

  Used as a lighter background color for UI elements like status bars.
  Used as semi-light background depending on the theme shading design.

  Markup:
  <div style="background-color:#e5e9f0; width=60; height=60"></div>

  Styleguide Nord - Snow Storm
  */
  --nord5: #e5e9f0;

  /*
  Lighter shade color of the base component color.

  Used for punctuations, carets and structuring characters like curly- and square brackets.
  In the UI scope used as background, selection- and highlight color depending on the theme shading design.

  Markup:
  <div style="background-color:#eceff4; width=60; height=60"></div>

  Styleguide Nord - Snow Storm
  */
  --nord6: #eceff4;

  /*
  Bluish core color.

  Used for classes, types and documentation tags.

  Markup:
  <div style="background-color:#8fbcbb; width=60; height=60"></div>

  Styleguide Nord - Frost
  */
  --nord7: #8fbcbb;

  /*
  Bluish core accent color.

  Represents the accent color of the color palette.
  Main color for primary UI elements and methods/functions.

  Can be used for
    - Markup quotes
    - Markup link URLs

  Markup:
  <div style="background-color:#88c0d0; width=60; height=60"></div>

  Styleguide Nord - Frost
  */
  --nord8: #88c0d0;

  /*
  Bluish core color.

  Used for language-specific syntactic/reserved support characters and keywords, operators, tags, units and
  punctuations like (semi)colons,commas and braces.

  Markup:
  <div style="background-color:#81a1c1; width=60; height=60"></div>

  Styleguide Nord - Frost
  */
  --nord9: #81a1c1;

  /*
  Bluish core color.

  Used for markup doctypes, import/include/require statements, pre-processor statements and at-rules (`@`).

  Markup:
  <div style="background-color:#5e81ac; width=60; height=60"></div>

  Styleguide Nord - Frost
  */
  --nord10: #5e81ac;

  /*
  Colorful component color.

  Used for errors, git/diff deletion and linter marker.

  Markup:
  <div style="background-color:#bf616a; width=60; height=60"></div>

  Styleguide Nord - Aurora
  */
  --nord11: #bf616a;

  /*
  Colorful component color.

  Used for annotations.

  Markup:
  <div style="background-color:#d08770; width=60; height=60"></div>

  Styleguide Nord - Aurora
  */
  --nord12: #d08770;

  /*
  Colorful component color.

  Used for escape characters, regular expressions and markup entities.
  In the UI scope used for warnings and git/diff renamings.

  Markup:
  <div style="background-color:#ebcb8b; width=60; height=60"></div>

  Styleguide Nord - Aurora
  */
  --nord13: #ebcb8b;

  /*
  Colorful component color.

  Main color for strings and attribute values.
  In the UI scope used for git/diff additions and success visualizations.

  Markup:
  <div style="background-color:#a3be8c; width=60; height=60"></div>

  Styleguide Nord - Aurora
  */
  --nord14: #a3be8c;

  /*
  Colorful component color.

  Used for numbers.

  Markup:
  <div style="background-color:#b48ead; width=60; height=60"></div>

  Styleguide Nord - Aurora
  */
  --nord15: #b48ead;
}

"""

catppuccin = """
{
    "version": "1.7.1",
    "latte": {
      "name": "Latte",
      "emoji": "🌻",
      "order": 0,
      "dark": false,
      "colors": {
        "rosewater": {
          "name": "Rosewater",
          "order": 0,
          "hex": "#dc8a78",
          "rgb": {
            "r": 220,
            "g": 138,
            "b": 120
          },
          "hsl": {
            "h": 10.799999999999995,
            "s": 0.5882352941176472,
            "l": 0.6666666666666667
          },
          "accent": true
        },
        "flamingo": {
          "name": "Flamingo",
          "order": 1,
          "hex": "#dd7878",
          "rgb": {
            "r": 221,
            "g": 120,
            "b": 120
          },
          "hsl": {
            "h": 0,
            "s": 0.5976331360946746,
            "l": 0.6686274509803922
          },
          "accent": true
        },
        "pink": {
          "name": "Pink",
          "order": 2,
          "hex": "#ea76cb",
          "rgb": {
            "r": 234,
            "g": 118,
            "b": 203
          },
          "hsl": {
            "h": 316.0344827586207,
            "s": 0.7341772151898731,
            "l": 0.6901960784313725
          },
          "accent": true
        },
        "mauve": {
          "name": "Mauve",
          "order": 3,
          "hex": "#8839ef",
          "rgb": {
            "r": 136,
            "g": 57,
            "b": 239
          },
          "hsl": {
            "h": 266.0439560439561,
            "s": 0.8504672897196262,
            "l": 0.5803921568627451
          },
          "accent": true
        },
        "red": {
          "name": "Red",
          "order": 4,
          "hex": "#d20f39",
          "rgb": {
            "r": 210,
            "g": 15,
            "b": 57
          },
          "hsl": {
            "h": 347.0769230769231,
            "s": 0.8666666666666666,
            "l": 0.4411764705882353
          },
          "accent": true
        },
        "maroon": {
          "name": "Maroon",
          "order": 5,
          "hex": "#e64553",
          "rgb": {
            "r": 230,
            "g": 69,
            "b": 83
          },
          "hsl": {
            "h": 354.78260869565213,
            "s": 0.76303317535545,
            "l": 0.5862745098039216
          },
          "accent": true
        },
        "peach": {
          "name": "Peach",
          "order": 6,
          "hex": "#fe640b",
          "rgb": {
            "r": 254,
            "g": 100,
            "b": 11
          },
          "hsl": {
            "h": 21.975308641975307,
            "s": 0.9918367346938776,
            "l": 0.5196078431372549
          },
          "accent": true
        },
        "yellow": {
          "name": "Yellow",
          "order": 7,
          "hex": "#df8e1d",
          "rgb": {
            "r": 223,
            "g": 142,
            "b": 29
          },
          "hsl": {
            "h": 34.948453608247426,
            "s": 0.7698412698412698,
            "l": 0.49411764705882355
          },
          "accent": true
        },
        "green": {
          "name": "Green",
          "order": 8,
          "hex": "#40a02b",
          "rgb": {
            "r": 64,
            "g": 160,
            "b": 43
          },
          "hsl": {
            "h": 109.23076923076923,
            "s": 0.5763546798029556,
            "l": 0.39803921568627454
          },
          "accent": true
        },
        "teal": {
          "name": "Teal",
          "order": 9,
          "hex": "#179299",
          "rgb": {
            "r": 23,
            "g": 146,
            "b": 153
          },
          "hsl": {
            "h": 183.23076923076923,
            "s": 0.7386363636363636,
            "l": 0.34509803921568627
          },
          "accent": true
        },
        "sky": {
          "name": "Sky",
          "order": 10,
          "hex": "#04a5e5",
          "rgb": {
            "r": 4,
            "g": 165,
            "b": 229
          },
          "hsl": {
            "h": 197.0666666666667,
            "s": 0.965665236051502,
            "l": 0.45686274509803926
          },
          "accent": true
        },
        "sapphire": {
          "name": "Sapphire",
          "order": 11,
          "hex": "#209fb5",
          "rgb": {
            "r": 32,
            "g": 159,
            "b": 181
          },
          "hsl": {
            "h": 188.85906040268458,
            "s": 0.6995305164319249,
            "l": 0.4176470588235294
          },
          "accent": true
        },
        "blue": {
          "name": "Blue",
          "order": 12,
          "hex": "#1e66f5",
          "rgb": {
            "r": 30,
            "g": 102,
            "b": 245
          },
          "hsl": {
            "h": 219.90697674418607,
            "s": 0.9148936170212768,
            "l": 0.5392156862745098
          },
          "accent": true
        },
        "lavender": {
          "name": "Lavender",
          "order": 13,
          "hex": "#7287fd",
          "rgb": {
            "r": 114,
            "g": 135,
            "b": 253
          },
          "hsl": {
            "h": 230.93525179856115,
            "s": 0.9720279720279721,
            "l": 0.7196078431372549
          },
          "accent": true
        },
        "text": {
          "name": "Text",
          "order": 14,
          "hex": "#4c4f69",
          "rgb": {
            "r": 76,
            "g": 79,
            "b": 105
          },
          "hsl": {
            "h": 233.79310344827587,
            "s": 0.16022099447513813,
            "l": 0.3549019607843137
          },
          "accent": false
        },
        "subtext1": {
          "name": "Subtext 1",
          "order": 15,
          "hex": "#5c5f77",
          "rgb": {
            "r": 92,
            "g": 95,
            "b": 119
          },
          "hsl": {
            "h": 233.33333333333334,
            "s": 0.1279620853080569,
            "l": 0.4137254901960784
          },
          "accent": false
        },
        "subtext0": {
          "name": "Subtext 0",
          "order": 16,
          "hex": "#6c6f85",
          "rgb": {
            "r": 108,
            "g": 111,
            "b": 133
          },
          "hsl": {
            "h": 232.79999999999998,
            "s": 0.10373443983402494,
            "l": 0.4725490196078431
          },
          "accent": false
        },
        "overlay2": {
          "name": "Overlay 2",
          "order": 17,
          "hex": "#7c7f93",
          "rgb": {
            "r": 124,
            "g": 127,
            "b": 147
          },
          "hsl": {
            "h": 232.17391304347825,
            "s": 0.09623430962343092,
            "l": 0.5313725490196078
          },
          "accent": false
        },
        "overlay1": {
          "name": "Overlay 1",
          "order": 18,
          "hex": "#8c8fa1",
          "rgb": {
            "r": 140,
            "g": 143,
            "b": 161
          },
          "hsl": {
            "h": 231.42857142857144,
            "s": 0.10047846889952144,
            "l": 0.5901960784313726
          },
          "accent": false
        },
        "overlay0": {
          "name": "Overlay 0",
          "order": 19,
          "hex": "#9ca0b0",
          "rgb": {
            "r": 156,
            "g": 160,
            "b": 176
          },
          "hsl": {
            "h": 228.00000000000003,
            "s": 0.11235955056179768,
            "l": 0.6509803921568628
          },
          "accent": false
        },
        "surface2": {
          "name": "Surface 2",
          "order": 20,
          "hex": "#acb0be",
          "rgb": {
            "r": 172,
            "g": 176,
            "b": 190
          },
          "hsl": {
            "h": 226.6666666666667,
            "s": 0.12162162162162159,
            "l": 0.7098039215686275
          },
          "accent": false
        },
        "surface1": {
          "name": "Surface 1",
          "order": 21,
          "hex": "#bcc0cc",
          "rgb": {
            "r": 188,
            "g": 192,
            "b": 204
          },
          "hsl": {
            "h": 225.00000000000003,
            "s": 0.13559322033898308,
            "l": 0.7686274509803922
          },
          "accent": false
        },
        "surface0": {
          "name": "Surface 0",
          "order": 22,
          "hex": "#ccd0da",
          "rgb": {
            "r": 204,
            "g": 208,
            "b": 218
          },
          "hsl": {
            "h": 222.85714285714292,
            "s": 0.1590909090909089,
            "l": 0.8274509803921568
          },
          "accent": false
        },
        "base": {
          "name": "Base",
          "order": 23,
          "hex": "#eff1f5",
          "rgb": {
            "r": 239,
            "g": 241,
            "b": 245
          },
          "hsl": {
            "h": 220.00000000000009,
            "s": 0.23076923076923136,
            "l": 0.9490196078431372
          },
          "accent": false
        },
        "mantle": {
          "name": "Mantle",
          "order": 24,
          "hex": "#e6e9ef",
          "rgb": {
            "r": 230,
            "g": 233,
            "b": 239
          },
          "hsl": {
            "h": 220.00000000000006,
            "s": 0.21951219512195116,
            "l": 0.919607843137255
          },
          "accent": false
        },
        "crust": {
          "name": "Crust",
          "order": 25,
          "hex": "#dce0e8",
          "rgb": {
            "r": 220,
            "g": 224,
            "b": 232
          },
          "hsl": {
            "h": 220.00000000000006,
            "s": 0.20689655172413762,
            "l": 0.8862745098039215
          },
          "accent": false
        }
      },
      "ansiColors": {
        "black": {
          "name": "Black",
          "order": 0,
          "normal": {
            "name": "Black",
            "hex": "#5c5f77",
            "rgb": {
              "r": 92,
              "g": 95,
              "b": 119
            },
            "hsl": {
              "h": 233.33333333333334,
              "s": 0.1279620853080569,
              "l": 0.4137254901960784
            },
            "code": 0
          },
          "bright": {
            "name": "Bright Black",
            "hex": "#6c6f85",
            "rgb": {
              "r": 108,
              "g": 111,
              "b": 133
            },
            "hsl": {
              "h": 232.79999999999998,
              "s": 0.10373443983402494,
              "l": 0.4725490196078431
            },
            "code": 8
          }
        },
        "red": {
          "name": "Red",
          "order": 1,
          "normal": {
            "name": "Red",
            "hex": "#d20f39",
            "rgb": {
              "r": 210,
              "g": 15,
              "b": 57
            },
            "hsl": {
              "h": 347.0769230769231,
              "s": 0.8666666666666666,
              "l": 0.4411764705882353
            },
            "code": 1
          },
          "bright": {
            "name": "Bright Red",
            "hex": "#de293e",
            "rgb": {
              "r": 222,
              "g": 41,
              "b": 62
            },
            "hsl": {
              "h": 353.0386740331492,
              "s": 0.7327935222672065,
              "l": 0.515686274509804
            },
            "code": 9
          }
        },
        "green": {
          "name": "Green",
          "order": 2,
          "normal": {
            "name": "Green",
            "hex": "#40a02b",
            "rgb": {
              "r": 64,
              "g": 160,
              "b": 43
            },
            "hsl": {
              "h": 109.23076923076923,
              "s": 0.5763546798029556,
              "l": 0.39803921568627454
            },
            "code": 2
          },
          "bright": {
            "name": "Bright Green",
            "hex": "#49af3d",
            "rgb": {
              "r": 73,
              "g": 175,
              "b": 61
            },
            "hsl": {
              "h": 113.68421052631581,
              "s": 0.48305084745762705,
              "l": 0.4627450980392157
            },
            "code": 10
          }
        },
        "yellow": {
          "name": "Yellow",
          "order": 3,
          "normal": {
            "name": "Yellow",
            "hex": "#df8e1d",
            "rgb": {
              "r": 223,
              "g": 142,
              "b": 29
            },
            "hsl": {
              "h": 34.948453608247426,
              "s": 0.7698412698412698,
              "l": 0.49411764705882355
            },
            "code": 3
          },
          "bright": {
            "name": "Bright Yellow",
            "hex": "#eea02d",
            "rgb": {
              "r": 238,
              "g": 160,
              "b": 45
            },
            "hsl": {
              "h": 35.751295336787564,
              "s": 0.8502202643171807,
              "l": 0.5549019607843138
            },
            "code": 11
          }
        },
        "blue": {
          "name": "Blue",
          "order": 4,
          "normal": {
            "name": "Blue",
            "hex": "#1e66f5",
            "rgb": {
              "r": 30,
              "g": 102,
              "b": 245
            },
            "hsl": {
              "h": 219.90697674418607,
              "s": 0.9148936170212768,
              "l": 0.5392156862745098
            },
            "code": 4
          },
          "bright": {
            "name": "Bright Blue",
            "hex": "#456eff",
            "rgb": {
              "r": 69,
              "g": 110,
              "b": 255
            },
            "hsl": {
              "h": 226.77419354838707,
              "s": 1,
              "l": 0.6352941176470588
            },
            "code": 12
          }
        },
        "magenta": {
          "name": "Magenta",
          "order": 5,
          "normal": {
            "name": "Magenta",
            "hex": "#ea76cb",
            "rgb": {
              "r": 234,
              "g": 118,
              "b": 203
            },
            "hsl": {
              "h": 316.0344827586207,
              "s": 0.7341772151898731,
              "l": 0.6901960784313725
            },
            "code": 5
          },
          "bright": {
            "name": "Bright Magenta",
            "hex": "#fe85d8",
            "rgb": {
              "r": 254,
              "g": 133,
              "b": 216
            },
            "hsl": {
              "h": 318.8429752066116,
              "s": 0.983739837398374,
              "l": 0.7588235294117647
            },
            "code": 13
          }
        },
        "cyan": {
          "name": "Cyan",
          "order": 6,
          "normal": {
            "name": "Cyan",
            "hex": "#179299",
            "rgb": {
              "r": 23,
              "g": 146,
              "b": 153
            },
            "hsl": {
              "h": 183.23076923076923,
              "s": 0.7386363636363636,
              "l": 0.34509803921568627
            },
            "code": 6
          },
          "bright": {
            "name": "Bright Cyan",
            "hex": "#2d9fa8",
            "rgb": {
              "r": 45,
              "g": 159,
              "b": 168
            },
            "hsl": {
              "h": 184.39024390243904,
              "s": 0.5774647887323943,
              "l": 0.4176470588235294
            },
            "code": 14
          }
        },
        "white": {
          "name": "White",
          "order": 7,
          "normal": {
            "name": "White",
            "hex": "#acb0be",
            "rgb": {
              "r": 172,
              "g": 176,
              "b": 190
            },
            "hsl": {
              "h": 226.6666666666667,
              "s": 0.12162162162162159,
              "l": 0.7098039215686275
            },
            "code": 7
          },
          "bright": {
            "name": "Bright White",
            "hex": "#bcc0cc",
            "rgb": {
              "r": 188,
              "g": 192,
              "b": 204
            },
            "hsl": {
              "h": 225.00000000000003,
              "s": 0.13559322033898308,
              "l": 0.7686274509803922
            },
            "code": 15
          }
        }
      }
    },
    "frappe": {
      "name": "Frappé",
      "emoji": "🪴",
      "order": 1,
      "dark": true,
      "colors": {
        "rosewater": {
          "name": "Rosewater",
          "order": 0,
          "hex": "#f2d5cf",
          "rgb": {
            "r": 242,
            "g": 213,
            "b": 207
          },
          "hsl": {
            "h": 10.2857142857143,
            "s": 0.5737704918032784,
            "l": 0.8803921568627451
          },
          "accent": true
        },
        "flamingo": {
          "name": "Flamingo",
          "order": 1,
          "hex": "#eebebe",
          "rgb": {
            "r": 238,
            "g": 190,
            "b": 190
          },
          "hsl": {
            "h": 0,
            "s": 0.5853658536585367,
            "l": 0.8392156862745098
          },
          "accent": true
        },
        "pink": {
          "name": "Pink",
          "order": 2,
          "hex": "#f4b8e4",
          "rgb": {
            "r": 244,
            "g": 184,
            "b": 228
          },
          "hsl": {
            "h": 316,
            "s": 0.7317073170731713,
            "l": 0.8392156862745098
          },
          "accent": true
        },
        "mauve": {
          "name": "Mauve",
          "order": 3,
          "hex": "#ca9ee6",
          "rgb": {
            "r": 202,
            "g": 158,
            "b": 230
          },
          "hsl": {
            "h": 276.66666666666663,
            "s": 0.5901639344262294,
            "l": 0.7607843137254902
          },
          "accent": true
        },
        "red": {
          "name": "Red",
          "order": 4,
          "hex": "#e78284",
          "rgb": {
            "r": 231,
            "g": 130,
            "b": 132
          },
          "hsl": {
            "h": 358.8118811881188,
            "s": 0.6778523489932885,
            "l": 0.7078431372549019
          },
          "accent": true
        },
        "maroon": {
          "name": "Maroon",
          "order": 5,
          "hex": "#ea999c",
          "rgb": {
            "r": 234,
            "g": 153,
            "b": 156
          },
          "hsl": {
            "h": 357.77777777777777,
            "s": 0.6585365853658534,
            "l": 0.7588235294117647
          },
          "accent": true
        },
        "peach": {
          "name": "Peach",
          "order": 6,
          "hex": "#ef9f76",
          "rgb": {
            "r": 239,
            "g": 159,
            "b": 118
          },
          "hsl": {
            "h": 20.33057851239669,
            "s": 0.7908496732026143,
            "l": 0.7
          },
          "accent": true
        },
        "yellow": {
          "name": "Yellow",
          "order": 7,
          "hex": "#e5c890",
          "rgb": {
            "r": 229,
            "g": 200,
            "b": 144
          },
          "hsl": {
            "h": 39.52941176470588,
            "s": 0.6204379562043796,
            "l": 0.7313725490196079
          },
          "accent": true
        },
        "green": {
          "name": "Green",
          "order": 8,
          "hex": "#a6d189",
          "rgb": {
            "r": 166,
            "g": 209,
            "b": 137
          },
          "hsl": {
            "h": 95.83333333333331,
            "s": 0.4390243902439024,
            "l": 0.6784313725490196
          },
          "accent": true
        },
        "teal": {
          "name": "Teal",
          "order": 9,
          "hex": "#81c8be",
          "rgb": {
            "r": 129,
            "g": 200,
            "b": 190
          },
          "hsl": {
            "h": 171.5492957746479,
            "s": 0.3922651933701657,
            "l": 0.6450980392156862
          },
          "accent": true
        },
        "sky": {
          "name": "Sky",
          "order": 10,
          "hex": "#99d1db",
          "rgb": {
            "r": 153,
            "g": 209,
            "b": 219
          },
          "hsl": {
            "h": 189.09090909090907,
            "s": 0.47826086956521735,
            "l": 0.7294117647058823
          },
          "accent": true
        },
        "sapphire": {
          "name": "Sapphire",
          "order": 11,
          "hex": "#85c1dc",
          "rgb": {
            "r": 133,
            "g": 193,
            "b": 220
          },
          "hsl": {
            "h": 198.62068965517244,
            "s": 0.5541401273885351,
            "l": 0.692156862745098
          },
          "accent": true
        },
        "blue": {
          "name": "Blue",
          "order": 12,
          "hex": "#8caaee",
          "rgb": {
            "r": 140,
            "g": 170,
            "b": 238
          },
          "hsl": {
            "h": 221.6326530612245,
            "s": 0.7424242424242424,
            "l": 0.7411764705882353
          },
          "accent": true
        },
        "lavender": {
          "name": "Lavender",
          "order": 13,
          "hex": "#babbf1",
          "rgb": {
            "r": 186,
            "g": 187,
            "b": 241
          },
          "hsl": {
            "h": 238.90909090909093,
            "s": 0.6626506024096385,
            "l": 0.8372549019607842
          },
          "accent": true
        },
        "text": {
          "name": "Text",
          "order": 14,
          "hex": "#c6d0f5",
          "rgb": {
            "r": 198,
            "g": 208,
            "b": 245
          },
          "hsl": {
            "h": 227.2340425531915,
            "s": 0.7014925373134333,
            "l": 0.8686274509803922
          },
          "accent": false
        },
        "subtext1": {
          "name": "Subtext 1",
          "order": 15,
          "hex": "#b5bfe2",
          "rgb": {
            "r": 181,
            "g": 191,
            "b": 226
          },
          "hsl": {
            "h": 226.66666666666669,
            "s": 0.43689320388349495,
            "l": 0.7980392156862746
          },
          "accent": false
        },
        "subtext0": {
          "name": "Subtext 0",
          "order": 16,
          "hex": "#a5adce",
          "rgb": {
            "r": 165,
            "g": 173,
            "b": 206
          },
          "hsl": {
            "h": 228.29268292682926,
            "s": 0.2949640287769784,
            "l": 0.7274509803921569
          },
          "accent": false
        },
        "overlay2": {
          "name": "Overlay 2",
          "order": 17,
          "hex": "#949cbb",
          "rgb": {
            "r": 148,
            "g": 156,
            "b": 187
          },
          "hsl": {
            "h": 227.69230769230768,
            "s": 0.22285714285714275,
            "l": 0.6568627450980392
          },
          "accent": false
        },
        "overlay1": {
          "name": "Overlay 1",
          "order": 18,
          "hex": "#838ba7",
          "rgb": {
            "r": 131,
            "g": 139,
            "b": 167
          },
          "hsl": {
            "h": 226.66666666666669,
            "s": 0.16981132075471703,
            "l": 0.584313725490196
          },
          "accent": false
        },
        "overlay0": {
          "name": "Overlay 0",
          "order": 19,
          "hex": "#737994",
          "rgb": {
            "r": 115,
            "g": 121,
            "b": 148
          },
          "hsl": {
            "h": 229.0909090909091,
            "s": 0.13360323886639683,
            "l": 0.515686274509804
          },
          "accent": false
        },
        "surface2": {
          "name": "Surface 2",
          "order": 20,
          "hex": "#626880",
          "rgb": {
            "r": 98,
            "g": 104,
            "b": 128
          },
          "hsl": {
            "h": 228.00000000000003,
            "s": 0.1327433628318584,
            "l": 0.44313725490196076
          },
          "accent": false
        },
        "surface1": {
          "name": "Surface 1",
          "order": 21,
          "hex": "#51576d",
          "rgb": {
            "r": 81,
            "g": 87,
            "b": 109
          },
          "hsl": {
            "h": 227.14285714285714,
            "s": 0.14736842105263157,
            "l": 0.37254901960784315
          },
          "accent": false
        },
        "surface0": {
          "name": "Surface 0",
          "order": 22,
          "hex": "#414559",
          "rgb": {
            "r": 65,
            "g": 69,
            "b": 89
          },
          "hsl": {
            "h": 230.00000000000003,
            "s": 0.15584415584415584,
            "l": 0.30196078431372547
          },
          "accent": false
        },
        "base": {
          "name": "Base",
          "order": 23,
          "hex": "#303446",
          "rgb": {
            "r": 48,
            "g": 52,
            "b": 70
          },
          "hsl": {
            "h": 229.0909090909091,
            "s": 0.18644067796610175,
            "l": 0.23137254901960785
          },
          "accent": false
        },
        "mantle": {
          "name": "Mantle",
          "order": 24,
          "hex": "#292c3c",
          "rgb": {
            "r": 41,
            "g": 44,
            "b": 60
          },
          "hsl": {
            "h": 230.52631578947367,
            "s": 0.18811881188118806,
            "l": 0.19803921568627453
          },
          "accent": false
        },
        "crust": {
          "name": "Crust",
          "order": 25,
          "hex": "#232634",
          "rgb": {
            "r": 35,
            "g": 38,
            "b": 52
          },
          "hsl": {
            "h": 229.41176470588238,
            "s": 0.19540229885057467,
            "l": 0.17058823529411765
          },
          "accent": false
        }
      },
      "ansiColors": {
        "black": {
          "name": "Black",
          "order": 0,
          "normal": {
            "name": "Black",
            "hex": "#51576d",
            "rgb": {
              "r": 81,
              "g": 87,
              "b": 109
            },
            "hsl": {
              "h": 227.14285714285714,
              "s": 0.14736842105263157,
              "l": 0.37254901960784315
            },
            "code": 0
          },
          "bright": {
            "name": "Bright Black",
            "hex": "#626880",
            "rgb": {
              "r": 98,
              "g": 104,
              "b": 128
            },
            "hsl": {
              "h": 228.00000000000003,
              "s": 0.1327433628318584,
              "l": 0.44313725490196076
            },
            "code": 8
          }
        },
        "red": {
          "name": "Red",
          "order": 1,
          "normal": {
            "name": "Red",
            "hex": "#e78284",
            "rgb": {
              "r": 231,
              "g": 130,
              "b": 132
            },
            "hsl": {
              "h": 358.8118811881188,
              "s": 0.6778523489932885,
              "l": 0.7078431372549019
            },
            "code": 1
          },
          "bright": {
            "name": "Bright Red",
            "hex": "#e67172",
            "rgb": {
              "r": 230,
              "g": 113,
              "b": 114
            },
            "hsl": {
              "h": 359.4871794871795,
              "s": 0.7005988023952096,
              "l": 0.6725490196078432
            },
            "code": 9
          }
        },
        "green": {
          "name": "Green",
          "order": 2,
          "normal": {
            "name": "Green",
            "hex": "#a6d189",
            "rgb": {
              "r": 166,
              "g": 209,
              "b": 137
            },
            "hsl": {
              "h": 95.83333333333331,
              "s": 0.4390243902439024,
              "l": 0.6784313725490196
            },
            "code": 2
          },
          "bright": {
            "name": "Bright Green",
            "hex": "#8ec772",
            "rgb": {
              "r": 142,
              "g": 199,
              "b": 114
            },
            "hsl": {
              "h": 100.23529411764706,
              "s": 0.431472081218274,
              "l": 0.6137254901960785
            },
            "code": 10
          }
        },
        "yellow": {
          "name": "Yellow",
          "order": 3,
          "normal": {
            "name": "Yellow",
            "hex": "#e5c890",
            "rgb": {
              "r": 229,
              "g": 200,
              "b": 144
            },
            "hsl": {
              "h": 39.52941176470588,
              "s": 0.6204379562043796,
              "l": 0.7313725490196079
            },
            "code": 3
          },
          "bright": {
            "name": "Bright Yellow",
            "hex": "#d9ba73",
            "rgb": {
              "r": 217,
              "g": 186,
              "b": 115
            },
            "hsl": {
              "h": 41.764705882352935,
              "s": 0.5730337078651685,
              "l": 0.6509803921568628
            },
            "code": 11
          }
        },
        "blue": {
          "name": "Blue",
          "order": 4,
          "normal": {
            "name": "Blue",
            "hex": "#8caaee",
            "rgb": {
              "r": 140,
              "g": 170,
              "b": 238
            },
            "hsl": {
              "h": 221.6326530612245,
              "s": 0.7424242424242424,
              "l": 0.7411764705882353
            },
            "code": 4
          },
          "bright": {
            "name": "Bright Blue",
            "hex": "#7b9ef0",
            "rgb": {
              "r": 123,
              "g": 158,
              "b": 240
            },
            "hsl": {
              "h": 222.05128205128207,
              "s": 0.7959183673469388,
              "l": 0.711764705882353
            },
            "code": 12
          }
        },
        "magenta": {
          "name": "Magenta",
          "order": 5,
          "normal": {
            "name": "Magenta",
            "hex": "#f4b8e4",
            "rgb": {
              "r": 244,
              "g": 184,
              "b": 228
            },
            "hsl": {
              "h": 316,
              "s": 0.7317073170731713,
              "l": 0.8392156862745098
            },
            "code": 5
          },
          "bright": {
            "name": "Bright Magenta",
            "hex": "#f2a4db",
            "rgb": {
              "r": 242,
              "g": 164,
              "b": 219
            },
            "hsl": {
              "h": 317.6923076923077,
              "s": 0.7499999999999998,
              "l": 0.7960784313725491
            },
            "code": 13
          }
        },
        "cyan": {
          "name": "Cyan",
          "order": 6,
          "normal": {
            "name": "Cyan",
            "hex": "#81c8be",
            "rgb": {
              "r": 129,
              "g": 200,
              "b": 190
            },
            "hsl": {
              "h": 171.5492957746479,
              "s": 0.3922651933701657,
              "l": 0.6450980392156862
            },
            "code": 6
          },
          "bright": {
            "name": "Bright Cyan",
            "hex": "#5abfb5",
            "rgb": {
              "r": 90,
              "g": 191,
              "b": 181
            },
            "hsl": {
              "h": 174.05940594059405,
              "s": 0.44104803493449785,
              "l": 0.5509803921568628
            },
            "code": 14
          }
        },
        "white": {
          "name": "White",
          "order": 7,
          "normal": {
            "name": "White",
            "hex": "#a5adce",
            "rgb": {
              "r": 165,
              "g": 173,
              "b": 206
            },
            "hsl": {
              "h": 228.29268292682926,
              "s": 0.2949640287769784,
              "l": 0.7274509803921569
            },
            "code": 7
          },
          "bright": {
            "name": "Bright White",
            "hex": "#b5bfe2",
            "rgb": {
              "r": 181,
              "g": 191,
              "b": 226
            },
            "hsl": {
              "h": 226.66666666666669,
              "s": 0.43689320388349495,
              "l": 0.7980392156862746
            },
            "code": 15
          }
        }
      }
    },
    "macchiato": {
      "name": "Macchiato",
      "emoji": "🌺",
      "order": 2,
      "dark": true,
      "colors": {
        "rosewater": {
          "name": "Rosewater",
          "order": 0,
          "hex": "#f4dbd6",
          "rgb": {
            "r": 244,
            "g": 219,
            "b": 214
          },
          "hsl": {
            "h": 9.999999999999963,
            "s": 0.5769230769230775,
            "l": 0.8980392156862745
          },
          "accent": true
        },
        "flamingo": {
          "name": "Flamingo",
          "order": 1,
          "hex": "#f0c6c6",
          "rgb": {
            "r": 240,
            "g": 198,
            "b": 198
          },
          "hsl": {
            "h": 0,
            "s": 0.5833333333333333,
            "l": 0.8588235294117648
          },
          "accent": true
        },
        "pink": {
          "name": "Pink",
          "order": 2,
          "hex": "#f5bde6",
          "rgb": {
            "r": 245,
            "g": 189,
            "b": 230
          },
          "hsl": {
            "h": 316.0714285714286,
            "s": 0.7368421052631583,
            "l": 0.8509803921568628
          },
          "accent": true
        },
        "mauve": {
          "name": "Mauve",
          "order": 3,
          "hex": "#c6a0f6",
          "rgb": {
            "r": 198,
            "g": 160,
            "b": 246
          },
          "hsl": {
            "h": 266.51162790697674,
            "s": 0.8269230769230772,
            "l": 0.7960784313725491
          },
          "accent": true
        },
        "red": {
          "name": "Red",
          "order": 4,
          "hex": "#ed8796",
          "rgb": {
            "r": 237,
            "g": 135,
            "b": 150
          },
          "hsl": {
            "h": 351.1764705882353,
            "s": 0.7391304347826088,
            "l": 0.7294117647058824
          },
          "accent": true
        },
        "maroon": {
          "name": "Maroon",
          "order": 5,
          "hex": "#ee99a0",
          "rgb": {
            "r": 238,
            "g": 153,
            "b": 160
          },
          "hsl": {
            "h": 355.05882352941177,
            "s": 0.7142857142857143,
            "l": 0.7666666666666666
          },
          "accent": true
        },
        "peach": {
          "name": "Peach",
          "order": 6,
          "hex": "#f5a97f",
          "rgb": {
            "r": 245,
            "g": 169,
            "b": 127
          },
          "hsl": {
            "h": 21.355932203389827,
            "s": 0.8550724637681162,
            "l": 0.7294117647058824
          },
          "accent": true
        },
        "yellow": {
          "name": "Yellow",
          "order": 7,
          "hex": "#eed49f",
          "rgb": {
            "r": 238,
            "g": 212,
            "b": 159
          },
          "hsl": {
            "h": 40.253164556962034,
            "s": 0.6991150442477877,
            "l": 0.7784313725490196
          },
          "accent": true
        },
        "green": {
          "name": "Green",
          "order": 8,
          "hex": "#a6da95",
          "rgb": {
            "r": 166,
            "g": 218,
            "b": 149
          },
          "hsl": {
            "h": 105.21739130434783,
            "s": 0.4825174825174825,
            "l": 0.7196078431372549
          },
          "accent": true
        },
        "teal": {
          "name": "Teal",
          "order": 9,
          "hex": "#8bd5ca",
          "rgb": {
            "r": 139,
            "g": 213,
            "b": 202
          },
          "hsl": {
            "h": 171.08108108108107,
            "s": 0.46835443037974706,
            "l": 0.6901960784313725
          },
          "accent": true
        },
        "sky": {
          "name": "Sky",
          "order": 10,
          "hex": "#91d7e3",
          "rgb": {
            "r": 145,
            "g": 215,
            "b": 227
          },
          "hsl": {
            "h": 188.78048780487802,
            "s": 0.5942028985507245,
            "l": 0.7294117647058823
          },
          "accent": true
        },
        "sapphire": {
          "name": "Sapphire",
          "order": 11,
          "hex": "#7dc4e4",
          "rgb": {
            "r": 125,
            "g": 196,
            "b": 228
          },
          "hsl": {
            "h": 198.64077669902912,
            "s": 0.6560509554140128,
            "l": 0.692156862745098
          },
          "accent": true
        },
        "blue": {
          "name": "Blue",
          "order": 12,
          "hex": "#8aadf4",
          "rgb": {
            "r": 138,
            "g": 173,
            "b": 244
          },
          "hsl": {
            "h": 220.188679245283,
            "s": 0.8281250000000003,
            "l": 0.7490196078431373
          },
          "accent": true
        },
        "lavender": {
          "name": "Lavender",
          "order": 13,
          "hex": "#b7bdf8",
          "rgb": {
            "r": 183,
            "g": 189,
            "b": 248
          },
          "hsl": {
            "h": 234.46153846153848,
            "s": 0.8227848101265824,
            "l": 0.8450980392156863
          },
          "accent": true
        },
        "text": {
          "name": "Text",
          "order": 14,
          "hex": "#cad3f5",
          "rgb": {
            "r": 202,
            "g": 211,
            "b": 245
          },
          "hsl": {
            "h": 227.4418604651163,
            "s": 0.6825396825396831,
            "l": 0.8764705882352941
          },
          "accent": false
        },
        "subtext1": {
          "name": "Subtext 1",
          "order": 15,
          "hex": "#b8c0e0",
          "rgb": {
            "r": 184,
            "g": 192,
            "b": 224
          },
          "hsl": {
            "h": 228,
            "s": 0.39215686274509803,
            "l": 0.8
          },
          "accent": false
        },
        "subtext0": {
          "name": "Subtext 0",
          "order": 16,
          "hex": "#a5adcb",
          "rgb": {
            "r": 165,
            "g": 173,
            "b": 203
          },
          "hsl": {
            "h": 227.36842105263156,
            "s": 0.2676056338028167,
            "l": 0.7215686274509804
          },
          "accent": false
        },
        "overlay2": {
          "name": "Overlay 2",
          "order": 17,
          "hex": "#939ab7",
          "rgb": {
            "r": 147,
            "g": 154,
            "b": 183
          },
          "hsl": {
            "h": 228.33333333333331,
            "s": 0.2000000000000001,
            "l": 0.6470588235294117
          },
          "accent": false
        },
        "overlay1": {
          "name": "Overlay 1",
          "order": 18,
          "hex": "#8087a2",
          "rgb": {
            "r": 128,
            "g": 135,
            "b": 162
          },
          "hsl": {
            "h": 227.6470588235294,
            "s": 0.1545454545454545,
            "l": 0.5686274509803921
          },
          "accent": false
        },
        "overlay0": {
          "name": "Overlay 0",
          "order": 19,
          "hex": "#6e738d",
          "rgb": {
            "r": 110,
            "g": 115,
            "b": 141
          },
          "hsl": {
            "h": 230.32258064516128,
            "s": 0.12350597609561753,
            "l": 0.49215686274509807
          },
          "accent": false
        },
        "surface2": {
          "name": "Surface 2",
          "order": 20,
          "hex": "#5b6078",
          "rgb": {
            "r": 91,
            "g": 96,
            "b": 120
          },
          "hsl": {
            "h": 229.65517241379308,
            "s": 0.13744075829383887,
            "l": 0.4137254901960784
          },
          "accent": false
        },
        "surface1": {
          "name": "Surface 1",
          "order": 21,
          "hex": "#494d64",
          "rgb": {
            "r": 73,
            "g": 77,
            "b": 100
          },
          "hsl": {
            "h": 231.11111111111114,
            "s": 0.15606936416184972,
            "l": 0.3392156862745098
          },
          "accent": false
        },
        "surface0": {
          "name": "Surface 0",
          "order": 22,
          "hex": "#363a4f",
          "rgb": {
            "r": 54,
            "g": 58,
            "b": 79
          },
          "hsl": {
            "h": 230.4,
            "s": 0.1879699248120301,
            "l": 0.2607843137254902
          },
          "accent": false
        },
        "base": {
          "name": "Base",
          "order": 23,
          "hex": "#24273a",
          "rgb": {
            "r": 36,
            "g": 39,
            "b": 58
          },
          "hsl": {
            "h": 231.8181818181818,
            "s": 0.23404255319148934,
            "l": 0.1843137254901961
          },
          "accent": false
        },
        "mantle": {
          "name": "Mantle",
          "order": 24,
          "hex": "#1e2030",
          "rgb": {
            "r": 30,
            "g": 32,
            "b": 48
          },
          "hsl": {
            "h": 233.33333333333334,
            "s": 0.23076923076923075,
            "l": 0.15294117647058825
          },
          "accent": false
        },
        "crust": {
          "name": "Crust",
          "order": 25,
          "hex": "#181926",
          "rgb": {
            "r": 24,
            "g": 25,
            "b": 38
          },
          "hsl": {
            "h": 235.71428571428572,
            "s": 0.22580645161290322,
            "l": 0.12156862745098039
          },
          "accent": false
        }
      },
      "ansiColors": {
        "black": {
          "name": "Black",
          "order": 0,
          "normal": {
            "name": "Black",
            "hex": "#494d64",
            "rgb": {
              "r": 73,
              "g": 77,
              "b": 100
            },
            "hsl": {
              "h": 231.11111111111114,
              "s": 0.15606936416184972,
              "l": 0.3392156862745098
            },
            "code": 0
          },
          "bright": {
            "name": "Bright Black",
            "hex": "#5b6078",
            "rgb": {
              "r": 91,
              "g": 96,
              "b": 120
            },
            "hsl": {
              "h": 229.65517241379308,
              "s": 0.13744075829383887,
              "l": 0.4137254901960784
            },
            "code": 8
          }
        },
        "red": {
          "name": "Red",
          "order": 1,
          "normal": {
            "name": "Red",
            "hex": "#ed8796",
            "rgb": {
              "r": 237,
              "g": 135,
              "b": 150
            },
            "hsl": {
              "h": 351.1764705882353,
              "s": 0.7391304347826088,
              "l": 0.7294117647058824
            },
            "code": 1
          },
          "bright": {
            "name": "Bright Red",
            "hex": "#ec7486",
            "rgb": {
              "r": 236,
              "g": 116,
              "b": 134
            },
            "hsl": {
              "h": 351,
              "s": 0.759493670886076,
              "l": 0.6901960784313725
            },
            "code": 9
          }
        },
        "green": {
          "name": "Green",
          "order": 2,
          "normal": {
            "name": "Green",
            "hex": "#a6da95",
            "rgb": {
              "r": 166,
              "g": 218,
              "b": 149
            },
            "hsl": {
              "h": 105.21739130434783,
              "s": 0.4825174825174825,
              "l": 0.7196078431372549
            },
            "code": 2
          },
          "bright": {
            "name": "Bright Green",
            "hex": "#8ccf7f",
            "rgb": {
              "r": 140,
              "g": 207,
              "b": 127
            },
            "hsl": {
              "h": 110.24999999999999,
              "s": 0.45454545454545453,
              "l": 0.6549019607843137
            },
            "code": 10
          }
        },
        "yellow": {
          "name": "Yellow",
          "order": 3,
          "normal": {
            "name": "Yellow",
            "hex": "#eed49f",
            "rgb": {
              "r": 238,
              "g": 212,
              "b": 159
            },
            "hsl": {
              "h": 40.253164556962034,
              "s": 0.6991150442477877,
              "l": 0.7784313725490196
            },
            "code": 3
          },
          "bright": {
            "name": "Bright Yellow",
            "hex": "#e1c682",
            "rgb": {
              "r": 225,
              "g": 198,
              "b": 130
            },
            "hsl": {
              "h": 42.94736842105264,
              "s": 0.6129032258064515,
              "l": 0.696078431372549
            },
            "code": 11
          }
        },
        "blue": {
          "name": "Blue",
          "order": 4,
          "normal": {
            "name": "Blue",
            "hex": "#8aadf4",
            "rgb": {
              "r": 138,
              "g": 173,
              "b": 244
            },
            "hsl": {
              "h": 220.188679245283,
              "s": 0.8281250000000003,
              "l": 0.7490196078431373
            },
            "code": 4
          },
          "bright": {
            "name": "Bright Blue",
            "hex": "#78a1f6",
            "rgb": {
              "r": 120,
              "g": 161,
              "b": 246
            },
            "hsl": {
              "h": 220.47619047619048,
              "s": 0.8750000000000002,
              "l": 0.7176470588235294
            },
            "code": 12
          }
        },
        "magenta": {
          "name": "Magenta",
          "order": 5,
          "normal": {
            "name": "Magenta",
            "hex": "#f5bde6",
            "rgb": {
              "r": 245,
              "g": 189,
              "b": 230
            },
            "hsl": {
              "h": 316.0714285714286,
              "s": 0.7368421052631583,
              "l": 0.8509803921568628
            },
            "code": 5
          },
          "bright": {
            "name": "Bright Magenta",
            "hex": "#f2a9dd",
            "rgb": {
              "r": 242,
              "g": 169,
              "b": 221
            },
            "hsl": {
              "h": 317.26027397260276,
              "s": 0.7373737373737372,
              "l": 0.8058823529411765
            },
            "code": 13
          }
        },
        "cyan": {
          "name": "Cyan",
          "order": 6,
          "normal": {
            "name": "Cyan",
            "hex": "#8bd5ca",
            "rgb": {
              "r": 139,
              "g": 213,
              "b": 202
            },
            "hsl": {
              "h": 171.08108108108107,
              "s": 0.46835443037974706,
              "l": 0.6901960784313725
            },
            "code": 6
          },
          "bright": {
            "name": "Bright Cyan",
            "hex": "#63cbc0",
            "rgb": {
              "r": 99,
              "g": 203,
              "b": 192
            },
            "hsl": {
              "h": 173.65384615384616,
              "s": 0.4999999999999998,
              "l": 0.592156862745098
            },
            "code": 14
          }
        },
        "white": {
          "name": "White",
          "order": 7,
          "normal": {
            "name": "White",
            "hex": "#a5adcb",
            "rgb": {
              "r": 165,
              "g": 173,
              "b": 203
            },
            "hsl": {
              "h": 227.36842105263156,
              "s": 0.2676056338028167,
              "l": 0.7215686274509804
            },
            "code": 7
          },
          "bright": {
            "name": "Bright White",
            "hex": "#b8c0e0",
            "rgb": {
              "r": 184,
              "g": 192,
              "b": 224
            },
            "hsl": {
              "h": 228,
              "s": 0.39215686274509803,
              "l": 0.8
            },
            "code": 15
          }
        }
      }
    },
    "mocha": {
      "name": "Mocha",
      "emoji": "🌿",
      "order": 3,
      "dark": true,
      "colors": {
        "rosewater": {
          "name": "Rosewater",
          "order": 0,
          "hex": "#f5e0dc",
          "rgb": {
            "r": 245,
            "g": 224,
            "b": 220
          },
          "hsl": {
            "h": 9.599999999999968,
            "s": 0.555555555555556,
            "l": 0.911764705882353
          },
          "accent": true
        },
        "flamingo": {
          "name": "Flamingo",
          "order": 1,
          "hex": "#f2cdcd",
          "rgb": {
            "r": 242,
            "g": 205,
            "b": 205
          },
          "hsl": {
            "h": 0,
            "s": 0.587301587301587,
            "l": 0.8764705882352941
          },
          "accent": true
        },
        "pink": {
          "name": "Pink",
          "order": 2,
          "hex": "#f5c2e7",
          "rgb": {
            "r": 245,
            "g": 194,
            "b": 231
          },
          "hsl": {
            "h": 316.4705882352941,
            "s": 0.7183098591549301,
            "l": 0.8607843137254902
          },
          "accent": true
        },
        "mauve": {
          "name": "Mauve",
          "order": 3,
          "hex": "#cba6f7",
          "rgb": {
            "r": 203,
            "g": 166,
            "b": 247
          },
          "hsl": {
            "h": 267.4074074074074,
            "s": 0.8350515463917528,
            "l": 0.8098039215686275
          },
          "accent": true
        },
        "red": {
          "name": "Red",
          "order": 4,
          "hex": "#f38ba8",
          "rgb": {
            "r": 243,
            "g": 139,
            "b": 168
          },
          "hsl": {
            "h": 343.2692307692308,
            "s": 0.8124999999999998,
            "l": 0.7490196078431373
          },
          "accent": true
        },
        "maroon": {
          "name": "Maroon",
          "order": 5,
          "hex": "#eba0ac",
          "rgb": {
            "r": 235,
            "g": 160,
            "b": 172
          },
          "hsl": {
            "h": 350.4,
            "s": 0.6521739130434779,
            "l": 0.7745098039215685
          },
          "accent": true
        },
        "peach": {
          "name": "Peach",
          "order": 6,
          "hex": "#fab387",
          "rgb": {
            "r": 250,
            "g": 179,
            "b": 135
          },
          "hsl": {
            "h": 22.95652173913043,
            "s": 0.92,
            "l": 0.7549019607843137
          },
          "accent": true
        },
        "yellow": {
          "name": "Yellow",
          "order": 7,
          "hex": "#f9e2af",
          "rgb": {
            "r": 249,
            "g": 226,
            "b": 175
          },
          "hsl": {
            "h": 41.35135135135135,
            "s": 0.8604651162790699,
            "l": 0.8313725490196078
          },
          "accent": true
        },
        "green": {
          "name": "Green",
          "order": 8,
          "hex": "#a6e3a1",
          "rgb": {
            "r": 166,
            "g": 227,
            "b": 161
          },
          "hsl": {
            "h": 115.45454545454544,
            "s": 0.5409836065573769,
            "l": 0.7607843137254902
          },
          "accent": true
        },
        "teal": {
          "name": "Teal",
          "order": 9,
          "hex": "#94e2d5",
          "rgb": {
            "r": 148,
            "g": 226,
            "b": 213
          },
          "hsl": {
            "h": 170.00000000000003,
            "s": 0.5735294117647057,
            "l": 0.7333333333333334
          },
          "accent": true
        },
        "sky": {
          "name": "Sky",
          "order": 10,
          "hex": "#89dceb",
          "rgb": {
            "r": 137,
            "g": 220,
            "b": 235
          },
          "hsl": {
            "h": 189.18367346938774,
            "s": 0.7101449275362316,
            "l": 0.7294117647058823
          },
          "accent": true
        },
        "sapphire": {
          "name": "Sapphire",
          "order": 11,
          "hex": "#74c7ec",
          "rgb": {
            "r": 116,
            "g": 199,
            "b": 236
          },
          "hsl": {
            "h": 198.5,
            "s": 0.759493670886076,
            "l": 0.6901960784313725
          },
          "accent": true
        },
        "blue": {
          "name": "Blue",
          "order": 12,
          "hex": "#89b4fa",
          "rgb": {
            "r": 137,
            "g": 180,
            "b": 250
          },
          "hsl": {
            "h": 217.1681415929203,
            "s": 0.9186991869918699,
            "l": 0.7588235294117647
          },
          "accent": true
        },
        "lavender": {
          "name": "Lavender",
          "order": 13,
          "hex": "#b4befe",
          "rgb": {
            "r": 180,
            "g": 190,
            "b": 254
          },
          "hsl": {
            "h": 231.89189189189187,
            "s": 0.9736842105263159,
            "l": 0.8509803921568628
          },
          "accent": true
        },
        "text": {
          "name": "Text",
          "order": 14,
          "hex": "#cdd6f4",
          "rgb": {
            "r": 205,
            "g": 214,
            "b": 244
          },
          "hsl": {
            "h": 226.15384615384616,
            "s": 0.6393442622950825,
            "l": 0.8803921568627451
          },
          "accent": false
        },
        "subtext1": {
          "name": "Subtext 1",
          "order": 15,
          "hex": "#bac2de",
          "rgb": {
            "r": 186,
            "g": 194,
            "b": 222
          },
          "hsl": {
            "h": 226.66666666666669,
            "s": 0.35294117647058837,
            "l": 0.8
          },
          "accent": false
        },
        "subtext0": {
          "name": "Subtext 0",
          "order": 16,
          "hex": "#a6adc8",
          "rgb": {
            "r": 166,
            "g": 173,
            "b": 200
          },
          "hsl": {
            "h": 227.6470588235294,
            "s": 0.23611111111111102,
            "l": 0.7176470588235294
          },
          "accent": false
        },
        "overlay2": {
          "name": "Overlay 2",
          "order": 17,
          "hex": "#9399b2",
          "rgb": {
            "r": 147,
            "g": 153,
            "b": 178
          },
          "hsl": {
            "h": 228.38709677419354,
            "s": 0.16756756756756758,
            "l": 0.6372549019607843
          },
          "accent": false
        },
        "overlay1": {
          "name": "Overlay 1",
          "order": 18,
          "hex": "#7f849c",
          "rgb": {
            "r": 127,
            "g": 132,
            "b": 156
          },
          "hsl": {
            "h": 229.65517241379308,
            "s": 0.12775330396475776,
            "l": 0.5549019607843138
          },
          "accent": false
        },
        "overlay0": {
          "name": "Overlay 0",
          "order": 19,
          "hex": "#6c7086",
          "rgb": {
            "r": 108,
            "g": 112,
            "b": 134
          },
          "hsl": {
            "h": 230.7692307692308,
            "s": 0.10743801652892565,
            "l": 0.4745098039215686
          },
          "accent": false
        },
        "surface2": {
          "name": "Surface 2",
          "order": 20,
          "hex": "#585b70",
          "rgb": {
            "r": 88,
            "g": 91,
            "b": 112
          },
          "hsl": {
            "h": 232.5,
            "s": 0.12,
            "l": 0.39215686274509803
          },
          "accent": false
        },
        "surface1": {
          "name": "Surface 1",
          "order": 21,
          "hex": "#45475a",
          "rgb": {
            "r": 69,
            "g": 71,
            "b": 90
          },
          "hsl": {
            "h": 234.2857142857143,
            "s": 0.13207547169811326,
            "l": 0.31176470588235294
          },
          "accent": false
        },
        "surface0": {
          "name": "Surface 0",
          "order": 22,
          "hex": "#313244",
          "rgb": {
            "r": 49,
            "g": 50,
            "b": 68
          },
          "hsl": {
            "h": 236.84210526315792,
            "s": 0.16239316239316234,
            "l": 0.22941176470588237
          },
          "accent": false
        },
        "base": {
          "name": "Base",
          "order": 23,
          "hex": "#1e1e2e",
          "rgb": {
            "r": 30,
            "g": 30,
            "b": 46
          },
          "hsl": {
            "h": 240,
            "s": 0.21052631578947367,
            "l": 0.14901960784313725
          },
          "accent": false
        },
        "mantle": {
          "name": "Mantle",
          "order": 24,
          "hex": "#181825",
          "rgb": {
            "r": 24,
            "g": 24,
            "b": 37
          },
          "hsl": {
            "h": 240,
            "s": 0.2131147540983607,
            "l": 0.11960784313725491
          },
          "accent": false
        },
        "crust": {
          "name": "Crust",
          "order": 25,
          "hex": "#11111b",
          "rgb": {
            "r": 17,
            "g": 17,
            "b": 27
          },
          "hsl": {
            "h": 240,
            "s": 0.22727272727272727,
            "l": 0.08627450980392157
          },
          "accent": false
        }
      },
      "ansiColors": {
        "black": {
          "name": "Black",
          "order": 0,
          "normal": {
            "name": "Black",
            "hex": "#45475a",
            "rgb": {
              "r": 69,
              "g": 71,
              "b": 90
            },
            "hsl": {
              "h": 234.2857142857143,
              "s": 0.13207547169811326,
              "l": 0.31176470588235294
            },
            "code": 0
          },
          "bright": {
            "name": "Bright Black",
            "hex": "#585b70",
            "rgb": {
              "r": 88,
              "g": 91,
              "b": 112
            },
            "hsl": {
              "h": 232.5,
              "s": 0.12,
              "l": 0.39215686274509803
            },
            "code": 8
          }
        },
        "red": {
          "name": "Red",
          "order": 1,
          "normal": {
            "name": "Red",
            "hex": "#f38ba8",
            "rgb": {
              "r": 243,
              "g": 139,
              "b": 168
            },
            "hsl": {
              "h": 343.2692307692308,
              "s": 0.8124999999999998,
              "l": 0.7490196078431373
            },
            "code": 1
          },
          "bright": {
            "name": "Bright Red",
            "hex": "#f37799",
            "rgb": {
              "r": 243,
              "g": 119,
              "b": 153
            },
            "hsl": {
              "h": 343.54838709677415,
              "s": 0.8378378378378376,
              "l": 0.7098039215686274
            },
            "code": 9
          }
        },
        "green": {
          "name": "Green",
          "order": 2,
          "normal": {
            "name": "Green",
            "hex": "#a6e3a1",
            "rgb": {
              "r": 166,
              "g": 227,
              "b": 161
            },
            "hsl": {
              "h": 115.45454545454544,
              "s": 0.5409836065573769,
              "l": 0.7607843137254902
            },
            "code": 2
          },
          "bright": {
            "name": "Bright Green",
            "hex": "#89d88b",
            "rgb": {
              "r": 137,
              "g": 216,
              "b": 139
            },
            "hsl": {
              "h": 121.51898734177213,
              "s": 0.5031847133757963,
              "l": 0.692156862745098
            },
            "code": 10
          }
        },
        "yellow": {
          "name": "Yellow",
          "order": 3,
          "normal": {
            "name": "Yellow",
            "hex": "#f9e2af",
            "rgb": {
              "r": 249,
              "g": 226,
              "b": 175
            },
            "hsl": {
              "h": 41.35135135135135,
              "s": 0.8604651162790699,
              "l": 0.8313725490196078
            },
            "code": 3
          },
          "bright": {
            "name": "Bright Yellow",
            "hex": "#ebd391",
            "rgb": {
              "r": 235,
              "g": 211,
              "b": 145
            },
            "hsl": {
              "h": 44,
              "s": 0.692307692307692,
              "l": 0.7450980392156863
            },
            "code": 11
          }
        },
        "blue": {
          "name": "Blue",
          "order": 4,
          "normal": {
            "name": "Blue",
            "hex": "#89b4fa",
            "rgb": {
              "r": 137,
              "g": 180,
              "b": 250
            },
            "hsl": {
              "h": 217.1681415929203,
              "s": 0.9186991869918699,
              "l": 0.7588235294117647
            },
            "code": 4
          },
          "bright": {
            "name": "Bright Blue",
            "hex": "#74a8fc",
            "rgb": {
              "r": 116,
              "g": 168,
              "b": 252
            },
            "hsl": {
              "h": 217.05882352941174,
              "s": 0.9577464788732396,
              "l": 0.7215686274509804
            },
            "code": 12
          }
        },
        "magenta": {
          "name": "Magenta",
          "order": 5,
          "normal": {
            "name": "Magenta",
            "hex": "#f5c2e7",
            "rgb": {
              "r": 245,
              "g": 194,
              "b": 231
            },
            "hsl": {
              "h": 316.4705882352941,
              "s": 0.7183098591549301,
              "l": 0.8607843137254902
            },
            "code": 5
          },
          "bright": {
            "name": "Bright Magenta",
            "hex": "#f2aede",
            "rgb": {
              "r": 242,
              "g": 174,
              "b": 222
            },
            "hsl": {
              "h": 317.6470588235294,
              "s": 0.7234042553191488,
              "l": 0.8156862745098039
            },
            "code": 13
          }
        },
        "cyan": {
          "name": "Cyan",
          "order": 6,
          "normal": {
            "name": "Cyan",
            "hex": "#94e2d5",
            "rgb": {
              "r": 148,
              "g": 226,
              "b": 213
            },
            "hsl": {
              "h": 170.00000000000003,
              "s": 0.5735294117647057,
              "l": 0.7333333333333334
            },
            "code": 6
          },
          "bright": {
            "name": "Bright Cyan",
            "hex": "#6bd7ca",
            "rgb": {
              "r": 107,
              "g": 215,
              "b": 202
            },
            "hsl": {
              "h": 172.77777777777777,
              "s": 0.5744680851063831,
              "l": 0.6313725490196078
            },
            "code": 14
          }
        },
        "white": {
          "name": "White",
          "order": 7,
          "normal": {
            "name": "White",
            "hex": "#a6adc8",
            "rgb": {
              "r": 166,
              "g": 173,
              "b": 200
            },
            "hsl": {
              "h": 227.6470588235294,
              "s": 0.23611111111111102,
              "l": 0.7176470588235294
            },
            "code": 7
          },
          "bright": {
            "name": "Bright White",
            "hex": "#bac2de",
            "rgb": {
              "r": 186,
              "g": 194,
              "b": 222
            },
            "hsl": {
              "h": 226.66666666666669,
              "s": 0.35294117647058837,
              "l": 0.8
            },
            "code": 15
          }
        }
      }
    }
  }
"""

everforest_raw = '\n          \tHex \tIdentifier \tUsages\n#232A2E \t#232A2E \tbg_dim \tDimmed Background\n#2D353B \t#2D353B \tbg0 \tDefault Background, Line Numbers Background, Signs Background, Status Line Background (inactive), Tab Line Label (active)\n#343F44 \t#343F44 \tbg1 \tCursor Line Background, Color Columns, Closed Folds Background, Status Line Background (active), Tab Line Background\n#3D484D \t#3D484D \tbg2 \tPopup Menu Background, Floating Window Background, Window Toolbar Background\n#475258 \t#475258 \tbg3 \tList Chars, Special Keys, Tab Line Label Background (inactive)\n#4F585E \t#4F585E \tbg4 \tWindow Splits Separators, Whitespaces, Breaks\n#56635f \t#56635f \tbg5 \tNot currently used\n#543A48 \t#543A48 \tbg_visual \tVisual Selection\n#514045 \t#514045 \tbg_red \tDiff Deleted Line Background, Error Highlights\n#425047 \t#425047 \tbg_green \tDiff Added Line Background, Hint Highlights\n#3A515D \t#3A515D \tbg_blue \tDiff Changed Line Background, Info Highlights\n#4D4C43 \t#4D4C43 \tbg_yellow \tWarning Highlights\n#D3C6AA \t#D3C6AA \tfg \tDefault Foreground, Signs, [Treesitter: Constants, Variables, Function Parameters, Properties, Symbol Identifiers]\n#E67E80 \t#E67E80 \tred \tConditional Keywords, Loop Keywords, Exception Keywords, Inclusion Keywords, Uncategorised Keywords, Diff Deleted Signs, Error Messages, Error Signs\n#E69875 \t#E69875 \torange \tOperator Keywords, Operators, Labels, Storage Classes, Composite Types, Enumerated Types, Tags, Title, Debugging Statements\n#DBBC7F \t#DBBC7F \tyellow \tTypes, Special Characters, Warning Messages, Warning Signs, [Treesitter: Modules, Namespaces]\n#A7C080 \t#A7C080 \tgreen \tFunction Names, Method Names, Strings, Characters, Hint Messages, Hint Signs, Search Highlights, [Treesitter: Constructors, Function Calls, Built-In Functions, Macro Functions, String Escapes, Regex Literals, Tag Delimiters, Non-Structured Text]\n#83C092 \t#83C092 \taqua \tConstants, Macros, [Treesitter: Strings, Characters]\n#7FBBB3 \t#7FBBB3 \tblue \tIdentifiers, Uncategorised Special Symbols, Diff Changed Text Background, Info Messages, Info Signs, [Treesitter: Fields, Special Punctuation, Math Environments]\n#D699B6 \t#D699B6 \tpurple \tBooleans, Numbers, Preprocessors, [Treesitter: Built-In Constants, Built-In Variables, Macro-Defined Constants, Attributes/Annotations]\n#7A8478 \t#7A8478 \tgrey0 \tLine Numbers, Fold Columns, Concealed Text, Foreground UI Elements\n#859289 \t#859289 \tgrey1 \tComments, Punctuation Delimiters, Closed Folds, Ignored/Disabled, UI Borders, Status Line Text\n#9DA9A0 \t#9DA9A0 \tgrey2 \tCursor Line Number, Tab Line Label (inactive)\n#A7C080 \t#A7C080 \tstatusline1 \tMenu Selection Background, Tab Line Label Background (active), Status Line Mode Indicator\n#D3C6AA \t#D3C6AA \tstatusline2 \tStatus Line Mode Indicator\n#E67E80 \t#E67E80 \tstatusline3 \tStatus Line Mode Indicator\n'

catppuccin_palette = json.loads(catppuccin)


def catppuccin_rgb_palette(catppuccin_palette: dict, color_key: Literal["latte", "frappe", "macchiato", "mocha"], return_list: bool = True) -> dict:
    colors = catppuccin_palette[color_key]["colors"]
    rgb_colors: dict = {color["name"]: tuple(color["rgb"].values()) for color in colors.values()}
    
    if return_list:
        return [v for v in rgb_colors.values()]
    
    return rgb_colors

def hex_to_rgb(hexcol: str) -> tuple[int, int, int]:
    r = int.from_bytes(bytes.fromhex(hexcol[0:2]))
    g = int.from_bytes(bytes.fromhex(hexcol[2:4]))
    b = int.from_bytes(bytes.fromhex(hexcol[4:6]))
    return (r,g,b)

def parse_hex_css(hex_css: str) -> list[tuple[int, int, int]]:
    colors = re.compile("#.+;").findall(hex_css)
    colors = [hex_to_rgb(c[1:-1]) for c in colors]

    return colors

color_pattern = re.compile(r"#([0-9A-Fa-f]{6})")

def parse_hex_colors_from_raw(text: str) -> list[str]:
    return color_pattern.findall(text)

rose_pine_raw = """
/*
* Variant: Rosé Pine
* Maintainer: DankChoir
*/

@define-color base            #191724;
@define-color surface         #1f1d2e;
@define-color overlay         #26233a;

@define-color muted           #6e6a86;
@define-color subtle          #908caa;
@define-color text            #e0def4;

@define-color love            #eb6f92;
@define-color gold            #f6c177;
@define-color rose            #ebbcba;
@define-color pine            #31748f;
@define-color foam            #9ccfd8;
@define-color iris            #c4a7e7;

@define-color highlightLow    #21202e;
@define-color highlightMed    #403d52;
@define-color highlightHigh   #524f67;

"""
everblush_raw = """

palette = 0=#232a2d;
palette = 1=#e57474;
palette = 2=#8ccf7e;
palette = 3=#e5c76b;
palette = 4=#67b0e8;
palette = 5=#c47fd5;
palette = 6=#6cbfbf;
palette = 7=#b3b9b8;
palette = 8=#2d3437;
palette = 9=#ef7e7e;
palette = 10=#96d988;
palette = 11=#f4d67a;
palette = 12=#71baf2;
palette = 13=#ce89df;
palette = 14=#67cbe7;
palette = 15=#bdc3c2;
background = #141b1e;
foreground = #dadada;
cursor-color = #dadada;

* ADDING MORE LIGHTS *
@define-color foam            #9ccfd8;
cursor-text = #141b1e;
selection-background = #141b1e;
selection-foreground = #dadada;
nord6: #eceff4;
idk: #d3c6aa; 
uhhhh: #9DA9A0;
aoidsjfoisdaf: #7A8478;


* some more dark cols wouldnt hurt
dark: #232A2E;
greenishdarks
#2D353B; #475258;

boi ts so tuff


"""


tokyonight_raw = """
palette = 0=#15161e
palette = 1=#f7768e
palette = 2=#9ece6a
palette = 3=#e0af68
palette = 4=#7aa2f7
palette = 5=#bb9af7
palette = 6=#7dcfff
palette = 7=#a9b1d6
palette = 8=#414868
palette = 9=#f7768e
palette = 10=#9ece6a
palette = 11=#e0af68
palette = 12=#7aa2f7
palette = 13=#bb9af7
palette = 14=#7dcfff
palette = 15=#c0caf5
background = #1a1b26
foreground = #c0caf5
cursor-color = #c0caf5
cursor-text = #1a1b26
selection-background = #283457
selection-foreground = #c0caf5


palette = 0=#1b1d2b
palette = 1=#ff757f
palette = 2=#c3e88d
palette = 3=#ffc777
palette = 4=#82aaff
palette = 5=#c099ff
palette = 6=#86e1fc
palette = 7=#828bb8
palette = 8=#444a73
palette = 9=#ff757f
palette = 10=#c3e88d
palette = 11=#ffc777
palette = 12=#82aaff
palette = 13=#c099ff
palette = 14=#86e1fc
palette = 15=#c8d3f5
background = #222436
foreground = #c8d3f5
cursor-color = #c8d3f5
cursor-text = #222436
selection-background = #2d3f76
selection-foreground = #c8d3f5
palette = 0=#15161e
palette = 1=#f7768e
palette = 2=#9ece6a
palette = 3=#e0af68
palette = 4=#7aa2f7
palette = 5=#bb9af7
palette = 6=#7dcfff
palette = 7=#a9b1d6
palette = 8=#414868
palette = 9=#f7768e
palette = 10=#9ece6a
palette = 11=#e0af68
palette = 12=#7aa2f7
palette = 13=#bb9af7
palette = 14=#7dcfff
palette = 15=#c0caf5
background = #1a1b26
foreground = #c0caf5
cursor-color = #c0caf5
cursor-text = #15161e
selection-background = #33467c
selection-foreground = #c0caf5
"""

CATPPUCCIN_MACCHIATO = catppuccin_rgb_palette(catppuccin_palette, "macchiato")
CATPPUCCIN_FRAPPE = catppuccin_rgb_palette(catppuccin_palette, "frappe")
CATPPUCCIN_MOCHA = catppuccin_rgb_palette(catppuccin_palette, "latte")
CATPPUCCIN_LATTE = catppuccin_rgb_palette(catppuccin_palette, "macchiato")
CATPPUCCIN = CATPPUCCIN_MACCHIATO + CATPPUCCIN_FRAPPE + CATPPUCCIN_MOCHA + CATPPUCCIN_LATTE
GRUVBOX = parse_hex_css(gruvbox_palette)
NORD=parse_hex_css(nord_css)
EVERFOREST = [hex_to_rgb(col) for col in parse_hex_colors_from_raw(everforest_raw)]
ROSE_PINE = parse_hex_css(rose_pine_raw)

EVERBLUSH = parse_hex_css(everblush_raw)
TOKYONIGHT = [hex_to_rgb(c) for c in parse_hex_colors_from_raw(tokyonight_raw)]


schemes = {
    "catppuccin_macchiato": CATPPUCCIN_MACCHIATO,
    "catppuccin_frappe": CATPPUCCIN_FRAPPE,
    "catppuccin_mocha": CATPPUCCIN_MOCHA,
    "catppuccin_latte": CATPPUCCIN_LATTE,
    "catppuccin": CATPPUCCIN,
    "gruvbox": GRUVBOX,
    "nord": NORD,
    "everforest": EVERFOREST,
    "rose_pine": ROSE_PINE,
    "everblush": list(set(EVERFOREST + EVERBLUSH)),
    "tokyonight": TOKYONIGHT,
}





import numpy as np 
import matplotlib.pyplot as plt 

def plot_palette(palette: list[tuple[int, int, int]]):
    f, axes = plt.subplots(1, len(palette))
    for i in range(len(axes)):
        axes[i].set_axis_off()
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        a[:,:,:] = palette[i]
        axes[i].imshow(a)
    f.tight_layout(pad=0)




