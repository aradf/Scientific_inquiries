#include <cstdlib>
#include <regex>      // use regex_math
#include <iostream>   // std::cin and std::cout
#include <string>     // string.
/*
 * g++ -std=c++17 -g -o ./run_compute my_module_14.cpp
 */

// Scientific Computing Library
namespace scl
{
}

/*
 * Regular expression: a regular expression is a specific pattern that provides concise nad flexible menas
 * to "match" strings of text, such as particular characters, words, or patterns of charaters.  
 */

int main(int argc, char** argv)
{
   std::cout << "Hello world" << std::endl;
   std::string input_string = "";
   std::string final_string = "end";
   
   {


   
   while (true)
   {
      std::cin >> input_string;
      // std::regex e("abc", std::regex_constants::icase);

      // . means any character except for new line.
      // std::regex e("abc.", std::regex_constants::icase);

      // ? means zero or 1 preceding characters.
      // std::regex e("abc?", std::regex_constants::icase);

      // * means zero or more preceding characters.
      // std::regex e("abc*", std::regex_constants::icase);

      // + one or more preceding characters.
      // std::regex e("abc+", std::regex_constants::icase);

      // any character inside the square bracket [] can be matched.
      // std::regex e("ab[cd]", std::regex_constants::icase);

      // any character except for what is inside the square bracket [] can be matched.
      // std::regex e("ab[^cd]", std::regex_constants::icase);

      // exactly three character match with characters inside the square bracket [] can be matched.
      // std::regex e("ab[cd]{3}", std::regex_constants::icase);

      // three or more character match with characters inside the square bracket [] can be matched.
      // std::regex e("ab[cd]{3,}", std::regex_constants::icase);

      // three, four or five characters match with characters inside the square bracket [] can be matched.
      // std::regex e("ab[cd]{3,5}", std::regex_constants::icase);

      // either ab or de[fg] match with characters inside the square bracket [] can be matched.
      // std::regex e("ab|de[fg]", std::regex_constants::icase);

      // match either abc or de followed by '\]fg
      // std::regex e("abc|de[\]fg", std::regex_constants::icase);

      // match either groupd (abc).  Further more \1 means match group (abc) again.
      // std::regex e("(abc)de+\\1", std::regex_constants::icase);

      // match groupd (abc) first, match group (de+), match group (de+) match group (abc)
      // std::regex e("(ab)c(de+)\\2\\1", std::regex_constants::icase);

      // [[:w:]] word character: digit, number, or undersore
      // There is a list of character classes shall be recognized.  Look it up 
      // on www.cplusplus.com
      // + means one or more.
      // \. is the raw character .
      std::regex e("[[:w:]]+@[[:w:]]+\.com", std::regex_constants::icase);


      bool match = std::regex_match(input_string, e);
      std::cout << (match? "Matched" : "Not Matched") << std::endl;
      if (final_string.compare(input_string) == 0)
         break;
   }
   }

   {
      std::string input_string = "";
      while(true)
      {
         std::cin >> input_string;

         //'abc' appears any where in the string object.
         //std::regex e("abc.", std::regex_constants::icase);

         //'abc' appears in the beginning.
         //std::regex e("^abc.", std::regex_constants::icase);

         //'abc' appears at the end.
         std::regex e("^abc.$", std::regex_constants::icase);

         bool match = std::regex_search(input_string, e);
         std::cout << (match? "Found" : "Not Found") << std::endl;
         if (final_string.compare(input_string) == 0)
            break;

      }

   }

   {
      /*
       * C++ has Regular Expression Grammers:

         ECMAScript
         basic
         extended
         awk
         grep
         egrep.
       */
       std::string input_string = "";
       while(true)
       {
          std::cin >> input_string;
          //std::regex_constants:grep makes this different from Regular Expression 
          std::regex e("^abc.+", std::regex_constants::grep);
 
          bool match = std::regex_search(input_string, e);
          std::cout << (match? "Found" : "Not Found") << std::endl;
          if (final_string.compare(input_string) == 0)
             break;
      }

   }

   {
       std::string input_string = "";
       

       while(true)
       {
          std::cin >> input_string;
          
          std::smatch string_match; // typedef std::match_results<string>
          std::regex e("([[:w:]]+)@([[:w:]]+)\.com");
 
          bool found = std::regex_search(input_string, string_match, e);
          std::cout << "string_match.size()" << string_match.size() << std::endl;

          for (int n=0; n<string_match.size(); n++)
          {
            std::cout << "string_match " << n << "]:; str()=" << string_match[n].str() << std::endl;
          }
          std::cout << "string_match.prefix().str(): " << string_match.prefix().str() << std::endl;
          std::cout << "string_match.suffix().str(): " << string_match.suffix().str() << std::endl;
          if (final_string.compare(input_string) == 0)
             break;

      }

   }


   return 0;
}