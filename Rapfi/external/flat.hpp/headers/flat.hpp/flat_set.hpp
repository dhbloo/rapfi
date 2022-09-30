/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include "flat_fwd.hpp"

namespace flat_hpp
{
    template < typename Key
             , typename Compare
             , typename Container >
    class flat_set : private Compare {
        using base_type = Compare;
    public:
        using key_type = Key;
        using value_type = Key;

        using size_type = typename Container::size_type;
        using difference_type = typename Container::difference_type;

        using key_compare = Compare;
        using value_compare = Compare;
        using container_type = Container;

        using reference = typename Container::reference;
        using const_reference = typename Container::const_reference;
        using pointer = typename Container::pointer;
        using const_pointer = typename Container::const_pointer;

        using iterator = typename Container::const_iterator;
        using const_iterator = typename Container::const_iterator;
        using reverse_iterator = typename Container::const_reverse_iterator;
        using const_reverse_iterator = typename Container::const_reverse_iterator;
    public:
        flat_set()
            noexcept(std::is_nothrow_default_constructible_v<base_type>
                && std::is_nothrow_default_constructible_v<container_type>) {}

        explicit flat_set(const Compare& c)
        : base_type(c) {}

        template < typename Allocator >
        explicit flat_set(const Allocator& a)
        : data_(a) {}

        template < typename Allocator >
        flat_set(const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {}

        template < typename InputIter >
        flat_set(InputIter first, InputIter last) {
            from_range_(first, last);
        }

        template < typename InputIter >
        flat_set(sorted_range_t, InputIter first, InputIter last) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter >
        flat_set(sorted_unique_range_t, InputIter first, InputIter last) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter >
        flat_set(InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(first, last);
        }

        template < typename InputIter >
        flat_set(sorted_range_t, InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter >
        flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(sorted_range_t, InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_set(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_unique_range, first, last);
        }

        flat_set(std::initializer_list<value_type> ilist) {
            from_range_(ilist.begin(), ilist.end());
        }

        flat_set(sorted_range_t, std::initializer_list<value_type> ilist) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        flat_set(std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(ilist.begin(), ilist.end());
        }

        flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_set(flat_set&& other, const Allocator& a)
        : base_type(static_cast<base_type&&>(other))
        , data_(std::move(other.data_), a) {}

        template < typename Allocator >
        flat_set(const flat_set& other, const Allocator& a)
        : base_type(static_cast<const base_type&>(other))
        , data_(other.data_, a) {}

        flat_set(flat_set&& other) = default;
        flat_set(const flat_set& other) = default;

        flat_set& operator=(flat_set&& other) = default;
        flat_set& operator=(const flat_set& other) = default;

        flat_set& operator=(std::initializer_list<value_type> ilist) {
            flat_set(ilist).swap(*this);
            return *this;
        }

        iterator begin()
        noexcept(noexcept(std::declval<container_type&>().begin())) {
            return data_.begin();
        }

        const_iterator begin() const
        noexcept(noexcept(std::declval<const container_type&>().begin())) {
            return data_.begin();
        }

        const_iterator cbegin() const
        noexcept(noexcept(std::declval<const container_type&>().cbegin())) {
            return data_.cbegin();
        }

        iterator end()
        noexcept(noexcept(std::declval<container_type&>().end())) {
            return data_.end();
        }

        const_iterator end() const
        noexcept(noexcept(std::declval<const container_type&>().end())) {
            return data_.end();
        }

        const_iterator cend() const
        noexcept(noexcept(std::declval<const container_type&>().cend())) {
            return data_.cend();
        }

        reverse_iterator rbegin()
        noexcept(noexcept(std::declval<container_type&>().rbegin())) {
            return data_.rbegin();
        }

        const_reverse_iterator rbegin() const
        noexcept(noexcept(std::declval<const container_type&>().rbegin())) {
            return data_.rbegin();
        }

        const_reverse_iterator crbegin() const
        noexcept(noexcept(std::declval<const container_type&>().crbegin())) {
            return data_.crbegin();
        }

        reverse_iterator rend()
        noexcept(noexcept(std::declval<container_type&>().rend())) {
            return data_.rend();
        }

        const_reverse_iterator rend() const
        noexcept(noexcept(std::declval<const container_type&>().rend())) {
            return data_.rend();
        }

        const_reverse_iterator crend() const
        noexcept(noexcept(std::declval<const container_type&>().crend())) {
            return data_.crend();
        }

        bool empty() const
        noexcept(noexcept(std::declval<const container_type&>().empty())) {
            return data_.empty();
        }

        size_type size() const
        noexcept(noexcept(std::declval<const container_type&>().size())) {
            return data_.size();
        }

        size_type max_size() const
        noexcept(noexcept(std::declval<const container_type&>().max_size())) {
            return data_.max_size();
        }

        size_type capacity() const
        noexcept(noexcept(std::declval<const container_type&>().capacity())) {
            return data_.capacity();
        }

        void reserve(size_type ncapacity) {
            data_.reserve(ncapacity);
        }

        void shrink_to_fit() {
            data_.shrink_to_fit();
        }

        std::pair<iterator, bool> insert(value_type&& value) {
            const iterator iter = lower_bound(value);
            return iter == end() || this->operator()(value, *iter)
                ? std::make_pair(data_.insert(iter, std::move(value)), true)
                : std::make_pair(iter, false);
        }

        std::pair<iterator, bool> insert(const value_type& value) {
            const iterator iter = lower_bound(value);
            return iter == end() || this->operator()(value, *iter)
                ? std::make_pair(data_.insert(iter, value), true)
                : std::make_pair(iter, false);
        }

        iterator insert(const_iterator hint, value_type&& value) {
            return (hint == begin() || this->operator()(*(hint - 1), value))
                && (hint == end() || this->operator()(value, *hint))
                ? data_.insert(hint, std::move(value))
                : insert(std::move(value)).first;
        }

        iterator insert(const_iterator hint, const value_type& value) {
            return (hint == begin() || this->operator()(*(hint - 1), value))
                && (hint == end() || this->operator()(value, *hint))
                ? data_.insert(hint, value)
                : insert(value).first;
        }

        template < typename InputIter >
        void insert(InputIter first, InputIter last) {
            insert_range_(first, last);
        }

        template < typename InputIter >
        void insert(sorted_range_t, InputIter first, InputIter last) {
            insert_range_(sorted_range, first, last);
        }

        void insert(std::initializer_list<value_type> ilist) {
            insert_range_(ilist.begin(), ilist.end());
        }

        void insert(sorted_range_t, std::initializer_list<value_type> ilist) {
            insert_range_(sorted_range, ilist.begin(), ilist.end());
        }

        template < typename... Args >
        std::pair<iterator, bool> emplace(Args&&... args) {
            return insert(value_type(std::forward<Args>(args)...));
        }

        template < typename... Args >
        iterator emplace_hint(const_iterator hint, Args&&... args) {
            return insert(hint, value_type(std::forward<Args>(args)...));
        }

        void clear()
        noexcept(noexcept(std::declval<container_type&>().clear())) {
            data_.clear();
        }

        iterator erase(const_iterator iter) {
            return data_.erase(iter);
        }

        iterator erase(const_iterator first, const_iterator last) {
            return data_.erase(first, last);
        }

        size_type erase(const key_type& key) {
            const const_iterator iter = find(key);
            return iter != end()
                ? (erase(iter), 1)
                : 0;
        }

        void swap(flat_set& other)
            noexcept(std::is_nothrow_swappable_v<base_type>
                && std::is_nothrow_swappable_v<container_type>)
        {
            using std::swap;
            swap(
                static_cast<base_type&>(*this),
                static_cast<base_type&>(other));
            swap(data_, other.data_);
        }

        size_type count(const key_type& key) const {
            const const_iterator iter = find(key);
            return iter != end() ? 1 : 0;
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            size_type>
        count(const K& key) const {
            const const_iterator iter = find(key);
            return iter != end() ? 1 : 0;
        }

        iterator find(const key_type& key) {
            const iterator iter = lower_bound(key);
            return iter != end() && !this->operator()(key, *iter)
                ? iter
                : end();
        }

        const_iterator find(const key_type& key) const {
            const const_iterator iter = lower_bound(key);
            return iter != end() && !this->operator()(key, *iter)
                ? iter
                : end();
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            iterator>
        find(const K& key) {
            const iterator iter = lower_bound(key);
            return iter != end() && !this->operator()(key, *iter)
                ? iter
                : end();
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const_iterator>
        find(const K& key) const {
            const const_iterator iter = lower_bound(key);
            return iter != end() && !this->operator()(key, *iter)
                ? iter
                : end();
        }

        bool contains(const key_type& key) const {
            return find(key) != end();
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            bool>
        contains(const K& key) const {
            return find(key) != end();
        }

        std::pair<iterator, iterator> equal_range(const key_type& key) {
            return std::equal_range(begin(), end(), key, key_comp());
        }

        std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
            return std::equal_range(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            std::pair<iterator, iterator>>
        equal_range(const K& key) {
            return std::equal_range(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            std::pair<const_iterator, const_iterator>>
        equal_range(const K& key) const {
            return std::equal_range(begin(), end(), key, key_comp());
        }

        iterator lower_bound(const key_type& key) {
            return std::lower_bound(begin(), end(), key, key_comp());
        }

        const_iterator lower_bound(const key_type& key) const {
            return std::lower_bound(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            iterator>
        lower_bound(const K& key) {
            return std::lower_bound(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const_iterator>
        lower_bound(const K& key) const {
            return std::lower_bound(begin(), end(), key, key_comp());
        }

        iterator upper_bound(const key_type& key) {
            return std::upper_bound(begin(), end(), key, key_comp());
        }

        const_iterator upper_bound(const key_type& key) const {
            return std::upper_bound(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            iterator>
        upper_bound(const K& key) {
            return std::upper_bound(begin(), end(), key, key_comp());
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const_iterator>
        upper_bound(const K& key) const {
            return std::upper_bound(begin(), end(), key, key_comp());
        }

        key_compare key_comp() const {
            return *this;
        }

        value_compare value_comp() const {
            return value_compare(key_comp());
        }
    private:
        template < typename Iter >
        void from_range_(Iter first, Iter last) {
            assert(data_.empty());
            data_.insert(data_.end(), first, last);
            std::sort(data_.begin(), data_.end(), key_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<key_compare>(key_comp())),
                data_.end());
        }

        template < typename Iter >
        void from_range_(sorted_range_t, Iter first, Iter last) {
            assert(data_.empty());
            assert(detail::is_sorted(first, last, key_comp()));
            data_.insert(data_.end(), first, last);
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<key_compare>(key_comp())),
                data_.end());
        }

        template < typename Iter >
        void from_range_(sorted_unique_range_t, Iter first, Iter last) {
            assert(data_.empty());
            assert(detail::is_sorted_unique(first, last, key_comp()));
            data_.insert(data_.end(), first, last);
        }
    private:
        template < typename Iter >
        void insert_range_(Iter first, Iter last) {
            const auto mid_iter = data_.insert(data_.end(), first, last);
            std::sort(mid_iter, data_.end(), key_comp());
            std::inplace_merge(data_.begin(), mid_iter, data_.end(), key_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<key_compare>(key_comp())),
                data_.end());
        }

        template < typename Iter >
        void insert_range_(sorted_range_t, Iter first, Iter last) {
            assert(detail::is_sorted(first, last, key_comp()));
            const auto mid_iter = data_.insert(data_.end(), first, last);
            std::inplace_merge(data_.begin(), mid_iter, data_.end(), key_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<key_compare>(key_comp())),
                data_.end());
        }
    private:
        container_type data_;
    };
}

namespace flat_hpp
{
    template < typename InputIter, typename Allocator
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<detail::iter_value_type<InputIter>>;

    template < typename InputIter, typename Allocator
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_range_t, InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<detail::iter_value_type<InputIter>>;

    template < typename InputIter, typename Allocator
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_unique_range_t, InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<detail::iter_value_type<InputIter>>;

    //

    template < typename InputIter
             , typename Compare = std::less<detail::iter_value_type<InputIter>>
             , typename Allocator = std::allocator<detail::iter_value_type<InputIter>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_set<detail::iter_value_type<InputIter>, Compare>;

    template < typename InputIter
             , typename Compare = std::less<detail::iter_value_type<InputIter>>
             , typename Allocator = std::allocator<detail::iter_value_type<InputIter>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_range_t, InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_set<detail::iter_value_type<InputIter>, Compare>;

    template < typename InputIter
             , typename Compare = std::less<detail::iter_value_type<InputIter>>
             , typename Allocator = std::allocator<detail::iter_value_type<InputIter>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_unique_range_t, InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_set<detail::iter_value_type<InputIter>, Compare>;

    //

    template < typename Key
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(std::initializer_list<Key>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<Key>;

    template < typename Key
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_range_t, std::initializer_list<Key>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<Key>;

    template < typename Key
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_unique_range_t, std::initializer_list<Key>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_set<Key>;

    //

    template < typename Key
             , typename Compare = std::less<Key>
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(std::initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
        -> flat_set<Key, Compare>;

    template < typename Key
             , typename Compare = std::less<Key>
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_range_t, std::initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
        -> flat_set<Key, Compare>;

    template < typename Key
             , typename Compare = std::less<Key>
             , typename Allocator = std::allocator<Key>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_set(sorted_unique_range_t, std::initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
        -> flat_set<Key, Compare>;
}

namespace flat_hpp
{
    template < typename Key
             , typename Compare
             , typename Container >
    void swap(
        flat_set<Key, Compare, Container>& l,
        flat_set<Key, Compare, Container>& r)
        noexcept(noexcept(l.swap(r)))
    {
        l.swap(r);
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator==(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return l.size() == r.size()
            && std::equal(l.begin(), l.end(), r.begin());
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator!=(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return !(l == r);
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator<(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return std::lexicographical_compare(l.begin(), l.end(), r.begin(), r.end());
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator>(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return r < l;
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator<=(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return !(r < l);
    }

    template < typename Key
             , typename Compare
             , typename Container >
    bool operator>=(
        const flat_set<Key, Compare, Container>& l,
        const flat_set<Key, Compare, Container>& r)
    {
        return !(l < r);
    }
}
