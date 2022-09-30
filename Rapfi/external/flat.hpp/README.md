# flat.hpp

> Library of flat vector-like based associative containers

[![linux][badge.linux]][linux]
[![darwin][badge.darwin]][darwin]
[![windows][badge.windows]][windows]
[![codecov][badge.codecov]][codecov]
[![language][badge.language]][language]
[![license][badge.license]][license]

[badge.darwin]: https://img.shields.io/github/workflow/status/BlackMATov/flat.hpp/darwin/main?label=Xcode&logo=xcode
[badge.linux]: https://img.shields.io/github/workflow/status/BlackMATov/flat.hpp/linux/main?label=GCC%2FClang&logo=linux
[badge.windows]: https://img.shields.io/github/workflow/status/BlackMATov/flat.hpp/windows/main?label=Visual%20Studio&logo=visual-studio
[badge.codecov]: https://img.shields.io/codecov/c/github/BlackMATov/flat.hpp/main?logo=codecov
[badge.language]: https://img.shields.io/badge/language-C%2B%2B17-yellow
[badge.license]: https://img.shields.io/badge/license-MIT-blue

[darwin]: https://github.com/BlackMATov/flat.hpp/actions?query=workflow%3Adarwin
[linux]: https://github.com/BlackMATov/flat.hpp/actions?query=workflow%3Alinux
[windows]: https://github.com/BlackMATov/flat.hpp/actions?query=workflow%3Awindows
[codecov]: https://codecov.io/gh/BlackMATov/flat.hpp
[language]: https://en.wikipedia.org/wiki/C%2B%2B17
[license]: https://en.wikipedia.org/wiki/MIT_License

[flat]: https://github.com/BlackMATov/flat.hpp

## Requirements

- [gcc](https://www.gnu.org/software/gcc/) **>= 7**
- [clang](https://clang.llvm.org/) **>= 5.0**
- [msvc](https://visualstudio.microsoft.com/) **>= 2017**

## Installation

[flat.hpp][flat] is a header-only library. All you need to do is copy the headers files from `headers` directory into your project and include them:

```cpp
#include "flat.hpp/flat_set.hpp"
using namespace flat_hpp;

int main() {
    flat_set<int> s;
    s.insert(42);
    return 0;
}
```

Also, you can add the root repository directory to your [cmake](https://cmake.org) project:

```cmake
add_subdirectory(external/flat.hpp)
target_link_libraries(your_project_target flat.hpp)
```

## API
- [Flat Set](#flat-set)
- [Flat Map](#flat-map)
- [Flat Multiset](#flat-multiset)
- [Flat Multimap](#flat-multimap)

## Flat Set

```cpp
template < typename Key
         , typename Compare = std::less<Key>
         , typename Container = std::vector<Key> >
class flat_set;
```

### Member types

| Member type              | Definition                          |
|--------------------------|-------------------------------------|
| `key_type`               | `Key`                               |
| `value_type`             | `Key`                               |
| `size_type`              | `Container::size_type`              |
| `difference_type`        | `Container::difference_type`        |
| `key_compare`            | `Compare`                           |
| `value_compare`          | `Compare`                           |
| `container_type`         | `Container`                         |
| `reference`              | `Container::reference`              |
| `const_reference`        | `Container::const_reference`        |
| `pointer`                | `Container::pointer`                |
| `const_pointer`          | `Container::const_pointer`          |
| `iterator`               | `Container::const_iterator`         |
| `const_iterator`         | `Container::const_iterator`         |
| `reverse_iterator`       | `Container::const_reverse_iterator` |
| `const_reverse_iterator` | `Container::const_reverse_iterator` |

### Member functions

```cpp
flat_set();

explicit flat_set(const Compare& c);

template < typename Allocator >
explicit flat_set(const Allocator& a);

template < typename Allocator >
flat_set(const Compare& c, const Allocator& a);

template < typename InputIter >
flat_set(InputIter first, InputIter last);
template < typename InputIter >
flat_set(sorted_range_t, InputIter first, InputIter last);
template < typename InputIter >
flat_set(sorted_unique_range_t, InputIter first, InputIter last);

template < typename InputIter >
flat_set(InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_set(sorted_range_t, InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c);

template < typename InputIter, typename Allocator >
flat_set(InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_set(sorted_range_t, InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Allocator& a);

template < typename InputIter, typename Allocator >
flat_set(InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_set(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);

flat_set(std::initializer_list<value_type> ilist);
flat_set(sorted_range_t, std::initializer_list<value_type> ilist);
flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist);

flat_set(std::initializer_list<value_type> ilist, const Compare& c);
flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c);
flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c);

template < typename Allocator >
flat_set(std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Allocator& a);

template < typename Allocator >
flat_set(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);

template < typename Allocator >
flat_set(flat_set&& other, const Allocator& a);

template < typename Allocator >
flat_set(const flat_set& other, const Allocator& a);

flat_set(flat_set&& other);
flat_set(const flat_set& other);

flat_set& operator=(flat_set&& other);
flat_set& operator=(const flat_set& other);

flat_set& operator=(std::initializer_list<value_type> ilist);
```

### Iterators

```cpp
iterator begin();
const_iterator begin() const;
const_iterator cbegin() const;

iterator end();
const_iterator end() const;
const_iterator cend() const;

reverse_iterator rbegin();
const_reverse_iterator rbegin() const;
const_reverse_iterator crbegin() const;

reverse_iterator rend();
const_reverse_iterator rend() const;
const_reverse_iterator crend() const;
```

### Capacity

```cpp
bool empty() const;
size_type size() const;
size_type max_size() const;
size_type capacity() const;
void reserve(size_type ncapacity);
void shrink_to_fit();
```

### Modifiers

```cpp
std::pair<iterator, bool> insert(value_type&& value);
std::pair<iterator, bool> insert(const value_type& value);

iterator insert(const_iterator hint, value_type&& value);
iterator insert(const_iterator hint, const value_type& value);

template < typename InputIter >
void insert(InputIter first, InputIter last);
template < typename InputIter >
void insert(sorted_range_t, InputIter first, InputIter last);

void insert(std::initializer_list<value_type> ilist);
void insert(sorted_range_t, std::initializer_list<value_type> ilist);

template < typename... Args >
std::pair<iterator, bool> emplace(Args&&... args);
template < typename... Args >
iterator emplace_hint(const_iterator hint, Args&&... args);

void clear();
iterator erase(const_iterator iter);
iterator erase(const_iterator first, const_iterator last);
size_type erase(const key_type& key);

void swap(flat_set& other);
```

### Lookup

```cpp
size_type count(const key_type& key) const;
template < typename K > size_type count(const K& key) const;

iterator find(const key_type& key);
const_iterator find(const key_type& key) const;

template < typename K > iterator find(const K& key);
template < typename K > const_iterator find(const K& key) const;

std::pair<iterator, iterator> equal_range(const key_type& key);
std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const;

template < typename K > std::pair<iterator, iterator> equal_range(const K& key);
template < typename K > std::pair<const_iterator, const_iterator> equal_range(const K& key) const;

iterator lower_bound(const key_type& key);
const_iterator lower_bound(const key_type& key) const;

template < typename K > iterator lower_bound(const K& key);
template < typename K > const_iterator lower_bound(const K& key) const;

iterator upper_bound(const key_type& key);
const_iterator upper_bound(const key_type& key) const;

template < typename K > iterator upper_bound(const K& key);
template < typename K > const_iterator upper_bound(const K& key) const;
```

### Observers

```cpp
key_compare key_comp() const;
value_compare value_comp() const;
```

### Non-member functions

```cpp
template < typename Key
         , typename Compare
         , typename Container >
void swap(
    flat_set<Key, Compare, Container>& l,
    flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator==(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator!=(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator<(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator>(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator<=(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator>=(
    const flat_set<Key, Compare, Container>& l,
    const flat_set<Key, Compare, Container>& r);
```

## Flat Map

```cpp
template < typename Key
         , typename Value
         , typename Compare = std::less<Key>
         , typename Container = std::vector<std::pair<Key, Value>> >
class flat_map;
```

### Member types

| Member type              | Definition                          |
|--------------------------|-------------------------------------|
| `key_type`               | `Key`                               |
| `mapped_type`            | `Value`                             |
| `value_type`             | `Container::value_type`             |
| `size_type`              | `Container::size_type`              |
| `difference_type`        | `Container::difference_type`        |
| `key_compare`            | `Compare`                           |
| `container_type`         | `Container`                         |
| `reference`              | `Container::reference`              |
| `const_reference`        | `Container::const_reference`        |
| `pointer`                | `Container::pointer`                |
| `const_pointer`          | `Container::const_pointer`          |
| `iterator`               | `Container::iterator`               |
| `const_iterator`         | `Container::const_iterator`         |
| `reverse_iterator`       | `Container::reverse_iterator`       |
| `const_reverse_iterator` | `Container::const_reverse_iterator` |

### Member classes

```cpp
class value_compare;
```

### Member functions

```cpp
flat_map();

explicit flat_map(const Compare& c);

template < typename Allocator >
explicit flat_map(const Allocator& a);

template < typename Allocator >
flat_map(const Compare& c, const Allocator& a);

template < typename InputIter >
flat_map(InputIter first, InputIter last);
template < typename InputIter >
flat_map(sorted_range_t, InputIter first, InputIter last);
template < typename InputIter >
flat_map(sorted_unique_range_t, InputIter first, InputIter last);

template < typename InputIter >
flat_map(InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_map(sorted_range_t, InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c);

template < typename InputIter, typename Allocator >
flat_map(InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_map(sorted_range_t, InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Allocator& a);

template < typename InputIter , typename Allocator >
flat_map(InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter , typename Allocator >
flat_map(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter , typename Allocator >
flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);

flat_map(std::initializer_list<value_type> ilist);
flat_map(sorted_range_t, std::initializer_list<value_type> ilist);
flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist);

flat_map(std::initializer_list<value_type> ilist, const Compare& c);
flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c);
flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c);

template < typename Allocator >
flat_map(std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Allocator& a);

template < typename Allocator >
flat_map(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);

template < typename Allocator >
flat_map(flat_map&& other, const Allocator& a);

template < typename Allocator >
flat_map(const flat_map& other, const Allocator& a);

flat_map(flat_map&& other);
flat_map(const flat_map& other);

flat_map& operator=(flat_map&& other);
flat_map& operator=(const flat_map& other);

flat_map& operator=(std::initializer_list<value_type> ilist);
```

### Iterators

```cpp
iterator begin();
const_iterator begin() const;
const_iterator cbegin() const;

iterator end();
const_iterator end() const;
const_iterator cend() const;

reverse_iterator rbegin();
const_reverse_iterator rbegin() const;
const_reverse_iterator crbegin() const;

reverse_iterator rend();
const_reverse_iterator rend() const;
const_reverse_iterator crend() const;
```

### Capacity

```cpp
bool empty() const;
size_type size() const;
size_type max_size() const;
size_type capacity() const;
void reserve(size_type ncapacity);
void shrink_to_fit();
```

### Element access

```cpp
mapped_type& operator[](key_type&& key);
mapped_type& operator[](const key_type& key);

mapped_type& at(const key_type& key);
const mapped_type& at(const key_type& key) const;

template < typename K > mapped_type& at(const K& key);
template < typename K > const mapped_type& at(const K& key) const;
```

### Modifiers

```cpp
std::pair<iterator, bool> insert(value_type&& value);
std::pair<iterator, bool> insert(const value_type& value);

iterator insert(const_iterator hint, value_type&& value);
iterator insert(const_iterator hint, const value_type& value);

template < typename TT >
std::pair<iterator, bool> insert_or_assign(key_type&& key, TT&& value);
template < typename TT >
std::pair<iterator, bool> insert_or_assign(const key_type& key, TT&& value);

template < typename InputIter >
void insert(InputIter first, InputIter last);
template < typename InputIter >
void insert(sorted_range_t, InputIter first, InputIter last);

void insert(std::initializer_list<value_type> ilist);
void insert(sorted_range_t, std::initializer_list<value_type> ilist);

template < typename... Args >
std::pair<iterator, bool> emplace(Args&&... args);
template < typename... Args >
iterator emplace_hint(const_iterator hint, Args&&... args);

template < typename... Args >
std::pair<iterator, bool> try_emplace(key_type&& key, Args&&... args);
template < typename... Args >
std::pair<iterator, bool> try_emplace(const key_type& key, Args&&... args);

void clear();
iterator erase(const_iterator iter);
iterator erase(const_iterator first, const_iterator last);
size_type erase(const key_type& key);

void swap(flat_map& other)
```

### Lookup

```cpp
size_type count(const key_type& key) const;
template < typename K > size_type count(const K& key) const;

iterator find(const key_type& key);
const_iterator find(const key_type& key) const;

template < typename K > iterator find(const K& key);
template < typename K > const_iterator find(const K& key) const;

std::pair<iterator, iterator> equal_range(const key_type& key);
std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const;

template < typename K > std::pair<iterator, iterator> equal_range(const K& key);
template < typename K > std::pair<const_iterator, const_iterator> equal_range(const K& key) const;

iterator lower_bound(const key_type& key);
const_iterator lower_bound(const key_type& key) const;

template < typename K > iterator lower_bound(const K& key);
template < typename K > const_iterator lower_bound(const K& key) const;

iterator upper_bound(const key_type& key);
const_iterator upper_bound(const key_type& key) const;

template < typename K > iterator upper_bound(const K& key);
template < typename K > const_iterator upper_bound(const K& key) const;
```

### Observers

```cpp
key_compare key_comp() const;
value_compare value_comp() const;
```

### Non-member functions

```cpp
template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
void swap(
    flat_map<Key, Value, Compare, Container>& l,
    flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator==(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator!=(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator<(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator>(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator<=(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator>=(
    const flat_map<Key, Value, Compare, Container>& l,
    const flat_map<Key, Value, Compare, Container>& r);
```

## Flat Multiset

```cpp
template < typename Key
         , typename Compare = std::less<Key>
         , typename Container = std::vector<Key> >
class flat_multiset;
```

### Member types

| Member type              | Definition                          |
|--------------------------|-------------------------------------|
| `key_type`               | `Key`                               |
| `value_type`             | `Key`                               |
| `size_type`              | `Container::size_type`              |
| `difference_type`        | `Container::difference_type`        |
| `key_compare`            | `Compare`                           |
| `value_compare`          | `Compare`                           |
| `container_type`         | `Container`                         |
| `reference`              | `Container::reference`              |
| `const_reference`        | `Container::const_reference`        |
| `pointer`                | `Container::pointer`                |
| `const_pointer`          | `Container::const_pointer`          |
| `iterator`               | `Container::const_iterator`         |
| `const_iterator`         | `Container::const_iterator`         |
| `reverse_iterator`       | `Container::const_reverse_iterator` |
| `const_reverse_iterator` | `Container::const_reverse_iterator` |

### Member functions

```cpp
flat_multiset();

explicit flat_multiset(const Compare& c);

template < typename Allocator >
explicit flat_multiset(const Allocator& a);

template < typename Allocator >
flat_multiset(const Compare& c, const Allocator& a);

template < typename InputIter >
flat_multiset(InputIter first, InputIter last);
template < typename InputIter >
flat_multiset(sorted_range_t, InputIter first, InputIter last);

template < typename InputIter >
flat_multiset(InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_multiset(sorted_range_t, InputIter first, InputIter last, const Compare& c);

template < typename InputIter, typename Allocator >
flat_multiset(InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_multiset(sorted_range_t, InputIter first, InputIter last, const Allocator& a);

template < typename InputIter, typename Allocator >
flat_multiset(InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_multiset(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);

flat_multiset(std::initializer_list<value_type> ilist);
flat_multiset(sorted_range_t, std::initializer_list<value_type> ilist);

flat_multiset(std::initializer_list<value_type> ilist, const Compare& c);
flat_multiset(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c);

template < typename Allocator >
flat_multiset(std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_multiset(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a);

template < typename Allocator >
flat_multiset(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_multiset(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);

template < typename Allocator >
flat_multiset(flat_multiset&& other, const Allocator& a);

template < typename Allocator >
flat_multiset(const flat_multiset& other, const Allocator& a);

flat_multiset(flat_multiset&& other);
flat_multiset(const flat_multiset& other);

flat_multiset& operator=(flat_multiset&& other);
flat_multiset& operator=(const flat_multiset& other);

flat_multiset& operator=(std::initializer_list<value_type> ilist);
```

### Iterators

```cpp
iterator begin();
const_iterator begin() const;
const_iterator cbegin() const;

iterator end();
const_iterator end() const;
const_iterator cend() const;

reverse_iterator rbegin();
const_reverse_iterator rbegin() const;
const_reverse_iterator crbegin() const;

reverse_iterator rend();
const_reverse_iterator rend() const;
const_reverse_iterator crend() const;
```

### Capacity

```cpp
bool empty() const;
size_type size() const;
size_type max_size() const;
size_type capacity() const;
void reserve(size_type ncapacity);
void shrink_to_fit();
```

### Modifiers

```cpp
iterator insert(value_type&& value);
iterator insert(const value_type& value);

iterator insert(const_iterator hint, value_type&& value);
iterator insert(const_iterator hint, const value_type& value);

template < typename InputIter >
void insert(InputIter first, InputIter last);
template < typename InputIter >
void insert(sorted_range_t, InputIter first, InputIter last);

void insert(std::initializer_list<value_type> ilist);
void insert(sorted_range_t, std::initializer_list<value_type> ilist);

template < typename... Args >
iterator emplace(Args&&... args);
template < typename... Args >
iterator emplace_hint(const_iterator hint, Args&&... args);

void clear();
iterator erase(const_iterator iter);
iterator erase(const_iterator first, const_iterator last);
size_type erase(const key_type& key);

void swap(flat_multiset& other);
```

### Lookup

```cpp
size_type count(const key_type& key) const;
template < typename K > size_type count(const K& key) const;

iterator find(const key_type& key);
const_iterator find(const key_type& key) const;

template < typename K > iterator find(const K& key);
template < typename K > const_iterator find(const K& key) const;

std::pair<iterator, iterator> equal_range(const key_type& key);
std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const;

template < typename K > std::pair<iterator, iterator> equal_range(const K& key);
template < typename K > std::pair<const_iterator, const_iterator> equal_range(const K& key) const;

iterator lower_bound(const key_type& key);
const_iterator lower_bound(const key_type& key) const;

template < typename K > iterator lower_bound(const K& key);
template < typename K > const_iterator lower_bound(const K& key) const;

iterator upper_bound(const key_type& key);
const_iterator upper_bound(const key_type& key) const;

template < typename K > iterator upper_bound(const K& key);
template < typename K > const_iterator upper_bound(const K& key) const;
```

### Observers

```cpp
key_compare key_comp() const;
value_compare value_comp() const;
```

### Non-member functions

```cpp
template < typename Key
         , typename Compare
         , typename Container >
void swap(
    flat_multiset<Key, Compare, Container>& l,
    flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator==(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator!=(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator<(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator>(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator<=(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);

template < typename Key
         , typename Compare
         , typename Container >
bool operator>=(
    const flat_multiset<Key, Compare, Container>& l,
    const flat_multiset<Key, Compare, Container>& r);
```

## Flat Multimap

```cpp
template < typename Key
         , typename Value
         , typename Compare = std::less<Key>
         , typename Container = std::vector<std::pair<Key, Value>> >
class flat_multimap;
```

### Member types

| Member type              | Definition                          |
|--------------------------|-------------------------------------|
| `key_type`               | `Key`                               |
| `mapped_type`            | `Value`                             |
| `value_type`             | `Container::value_type`             |
| `size_type`              | `Container::size_type`              |
| `difference_type`        | `Container::difference_type`        |
| `key_compare`            | `Compare`                           |
| `container_type`         | `Container`                         |
| `reference`              | `Container::reference`              |
| `const_reference`        | `Container::const_reference`        |
| `pointer`                | `Container::pointer`                |
| `const_pointer`          | `Container::const_pointer`          |
| `iterator`               | `Container::iterator`               |
| `const_iterator`         | `Container::const_iterator`         |
| `reverse_iterator`       | `Container::reverse_iterator`       |
| `const_reverse_iterator` | `Container::const_reverse_iterator` |

### Member classes

```cpp
class value_compare;
```

### Member functions

```cpp
flat_multimap();

explicit flat_multimap(const Compare& c);

template < typename Allocator >
explicit flat_multimap(const Allocator& a);

template < typename Allocator >
flat_multimap(const Compare& c, const Allocator& a);

template < typename InputIter >
flat_multimap(InputIter first, InputIter last);
template < typename InputIter >
flat_multimap(sorted_range_t, InputIter first, InputIter last);

template < typename InputIter >
flat_multimap(InputIter first, InputIter last, const Compare& c);
template < typename InputIter >
flat_multimap(sorted_range_t, InputIter first, InputIter last, const Compare& c);

template < typename InputIter, typename Allocator >
flat_multimap(InputIter first, InputIter last, const Allocator& a);
template < typename InputIter, typename Allocator >
flat_multimap(sorted_range_t, InputIter first, InputIter last, const Allocator& a);

template < typename InputIter , typename Allocator >
flat_multimap(InputIter first, InputIter last, const Compare& c, const Allocator& a);
template < typename InputIter , typename Allocator >
flat_multimap(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a);

flat_multimap(std::initializer_list<value_type> ilist);
flat_multimap(sorted_range_t, std::initializer_list<value_type> ilist);

flat_multimap(std::initializer_list<value_type> ilist, const Compare& c);
flat_multimap(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c);

template < typename Allocator >
flat_multimap(std::initializer_list<value_type> ilist, const Allocator& a);
template < typename Allocator >
flat_multimap(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a);

template < typename Allocator >
flat_multimap(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);
template < typename Allocator >
flat_multimap(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a);

template < typename Allocator >
flat_multimap(flat_multimap&& other, const Allocator& a);

template < typename Allocator >
flat_multimap(const flat_multimap& other, const Allocator& a);

flat_multimap(flat_multimap&& other);
flat_multimap(const flat_multimap& other);

flat_multimap& operator=(flat_multimap&& other);
flat_multimap& operator=(const flat_multimap& other);

flat_multimap& operator=(std::initializer_list<value_type> ilist);
```

### Iterators

```cpp
iterator begin();
const_iterator begin() const;
const_iterator cbegin() const;

iterator end();
const_iterator end() const;
const_iterator cend() const;

reverse_iterator rbegin();
const_reverse_iterator rbegin() const;
const_reverse_iterator crbegin() const;

reverse_iterator rend();
const_reverse_iterator rend() const;
const_reverse_iterator crend() const;
```

### Capacity

```cpp
bool empty() const;
size_type size() const;
size_type max_size() const;
size_type capacity() const;
void reserve(size_type ncapacity);
void shrink_to_fit();
```

### Element access

```cpp
mapped_type& operator[](key_type&& key);
mapped_type& operator[](const key_type& key);

mapped_type& at(const key_type& key);
const mapped_type& at(const key_type& key) const;

template < typename K > mapped_type& at(const K& key);
template < typename K > const mapped_type& at(const K& key) const;
```

### Modifiers

```cpp
iterator insert(value_type&& value);
iterator insert(const value_type& value);

iterator insert(const_iterator hint, value_type&& value);
iterator insert(const_iterator hint, const value_type& value);

template < typename InputIter >
void insert(InputIter first, InputIter last);
template < typename InputIter >
void insert(sorted_range_t, InputIter first, InputIter last);

void insert(std::initializer_list<value_type> ilist);
void insert(sorted_range_t, std::initializer_list<value_type> ilist);

template < typename... Args >
iterator emplace(Args&&... args);
template < typename... Args >
iterator emplace_hint(const_iterator hint, Args&&... args);

void clear();
iterator erase(const_iterator iter);
iterator erase(const_iterator first, const_iterator last);
size_type erase(const key_type& key);

void swap(flat_multimap& other)
```

### Lookup

```cpp
size_type count(const key_type& key) const;
template < typename K > size_type count(const K& key) const;

iterator find(const key_type& key);
const_iterator find(const key_type& key) const;

template < typename K > iterator find(const K& key);
template < typename K > const_iterator find(const K& key) const;

std::pair<iterator, iterator> equal_range(const key_type& key);
std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const;

template < typename K > std::pair<iterator, iterator> equal_range(const K& key);
template < typename K > std::pair<const_iterator, const_iterator> equal_range(const K& key) const;

iterator lower_bound(const key_type& key);
const_iterator lower_bound(const key_type& key) const;

template < typename K > iterator lower_bound(const K& key);
template < typename K > const_iterator lower_bound(const K& key) const;

iterator upper_bound(const key_type& key);
const_iterator upper_bound(const key_type& key) const;

template < typename K > iterator upper_bound(const K& key);
template < typename K > const_iterator upper_bound(const K& key) const;
```

### Observers

```cpp
key_compare key_comp() const;
value_compare value_comp() const;
```

### Non-member functions

```cpp
template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
void swap(
    flat_multimap<Key, Value, Compare, Container>& l,
    flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator==(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator!=(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator<(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator>(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator<=(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);

template < typename Key
         , typename Value
         , typename Compare
         , typename Container >
bool operator>=(
    const flat_multimap<Key, Value, Compare, Container>& l,
    const flat_multimap<Key, Value, Compare, Container>& r);
```

## [License (MIT)](./LICENSE.md)
