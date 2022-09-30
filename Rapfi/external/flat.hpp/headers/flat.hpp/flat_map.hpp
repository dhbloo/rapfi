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
             , typename Value
             , typename Compare
             , typename Container >
    class flat_map
        : private detail::pair_compare<
            typename Container::value_type,
            Compare>
    {
        using base_type = detail::pair_compare<
            typename Container::value_type,
            Compare>;
    public:
        using key_type = Key;
        using mapped_type = Value;
        using value_type = typename Container::value_type;

        using size_type = typename Container::size_type;
        using difference_type = typename Container::difference_type;

        using key_compare = Compare;
        using container_type = Container;

        using reference = typename Container::reference;
        using const_reference = typename Container::const_reference;
        using pointer = typename Container::pointer;
        using const_pointer = typename Container::const_pointer;

        using iterator = typename Container::iterator;
        using const_iterator = typename Container::const_iterator;
        using reverse_iterator = typename Container::reverse_iterator;
        using const_reverse_iterator = typename Container::const_reverse_iterator;

        class value_compare : private key_compare {
        public:
            bool operator()(const value_type& l, const value_type& r) const {
                return key_compare::operator()(l.first, r.first);
            }
        protected:
            friend class flat_map;
            explicit value_compare(key_compare compare)
            : key_compare(std::move(compare)) {}
        };
    public:
        flat_map()
            noexcept(std::is_nothrow_default_constructible_v<base_type>
                && std::is_nothrow_default_constructible_v<container_type>) {}

        explicit flat_map(const Compare& c)
        : base_type(c) {}

        template < typename Allocator >
        explicit flat_map(const Allocator& a)
        : data_(a) {}

        template < typename Allocator >
        flat_map(const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {}

        template < typename InputIter >
        flat_map(InputIter first, InputIter last) {
            from_range_(first, last);
        }

        template < typename InputIter >
        flat_map(sorted_range_t, InputIter first, InputIter last) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter >
        flat_map(sorted_unique_range_t, InputIter first, InputIter last) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter >
        flat_map(InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(first, last);
        }

        template < typename InputIter >
        flat_map(sorted_range_t, InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter >
        flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c)
        : base_type(c) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(sorted_range_t, InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Allocator& a)
        : data_(a) {
            from_range_(sorted_unique_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(sorted_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_range, first, last);
        }

        template < typename InputIter, typename Allocator >
        flat_map(sorted_unique_range_t, InputIter first, InputIter last, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_unique_range, first, last);
        }

        flat_map(std::initializer_list<value_type> ilist) {
            from_range_(ilist.begin(), ilist.end());
        }

        flat_map(sorted_range_t, std::initializer_list<value_type> ilist) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        flat_map(std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(ilist.begin(), ilist.end());
        }

        flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c)
        : base_type(c) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Allocator& a)
        : data_(a) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(sorted_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(sorted_unique_range_t, std::initializer_list<value_type> ilist, const Compare& c, const Allocator& a)
        : base_type(c)
        , data_(a) {
            from_range_(sorted_unique_range, ilist.begin(), ilist.end());
        }

        template < typename Allocator >
        flat_map(flat_map&& other, const Allocator& a)
        : base_type(static_cast<base_type&&>(other))
        , data_(std::move(other.data_), a) {}

        template < typename Allocator >
        flat_map(const flat_map& other, const Allocator& a)
        : base_type(static_cast<const base_type&>(other))
        , data_(other.data_, a) {}

        flat_map(flat_map&& other) = default;
        flat_map(const flat_map& other) = default;

        flat_map& operator=(flat_map&& other) = default;
        flat_map& operator=(const flat_map& other) = default;

        flat_map& operator=(std::initializer_list<value_type> ilist) {
            flat_map(ilist).swap(*this);
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

        mapped_type& operator[](key_type&& key) {
            const iterator iter = find(key);
            return iter != end()
                ? iter->second
                : emplace(std::move(key), mapped_type()).first->second;
        }

        mapped_type& operator[](const key_type& key) {
            const iterator iter = find(key);
            return iter != end()
                ? iter->second
                : emplace(key, mapped_type()).first->second;
        }

        mapped_type& at(const key_type& key) {
            const iterator iter = find(key);
            if ( iter != end() ) {
                return iter->second;
            }
            throw std::out_of_range("flat_map::at: key not found");
        }

        const mapped_type& at(const key_type& key) const {
            const const_iterator iter = find(key);
            if ( iter != end() ) {
                return iter->second;
            }
            throw std::out_of_range("flat_map::at: key not found");
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            mapped_type&>
        at(const K& key) {
            const iterator iter = find(key);
            if ( iter != end() ) {
                return iter->second;
            }
            throw std::out_of_range("flat_map::at: key not found");
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const mapped_type&>
        at(const K& key) const {
            const const_iterator iter = find(key);
            if ( iter != end() ) {
                return iter->second;
            }
            throw std::out_of_range("flat_map::at: key not found");
        }

        std::pair<iterator, bool> insert(value_type&& value) {
            const iterator iter = lower_bound(value.first);
            return iter == end() || this->operator()(value, *iter)
                ? std::make_pair(data_.insert(iter, std::move(value)), true)
                : std::make_pair(iter, false);
        }

        std::pair<iterator, bool> insert(const value_type& value) {
            const iterator iter = lower_bound(value.first);
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

        template < typename TT >
        std::pair<iterator, bool> insert_or_assign(key_type&& key, TT&& value) {
            iterator iter = lower_bound(key);
            if ( iter == end() || this->operator()(key, *iter) ) {
                iter = emplace_hint(iter, std::move(key), std::forward<TT>(value));
                return {iter, true};
            }
            (*iter).second = std::forward<TT>(value);
            return {iter, false};
        }

        template < typename TT >
        std::pair<iterator, bool> insert_or_assign(const key_type& key, TT&& value) {
            iterator iter = lower_bound(key);
            if ( iter == end() || this->operator()(key, *iter) ) {
                iter = emplace_hint(iter, key, std::forward<TT>(value));
                return {iter, true};
            }
            (*iter).second = std::forward<TT>(value);
            return {iter, false};
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

        template < typename... Args >
        std::pair<iterator, bool> try_emplace(key_type&& key, Args&&... args) {
            iterator iter = lower_bound(key);
            if ( iter == end() || this->operator()(key, *iter) ) {
                iter = emplace_hint(iter, std::move(key), std::forward<Args>(args)...);
                return {iter, true};
            }
            return {iter, false};
        }

        template < typename... Args >
        std::pair<iterator, bool> try_emplace(const key_type& key, Args&&... args) {
            iterator iter = lower_bound(key);
            if ( iter == end() || this->operator()(key, *iter) ) {
                iter = emplace_hint(iter, key, std::forward<Args>(args)...);
                return {iter, true};
            }
            return {iter, false};
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

        void swap(flat_map& other)
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
            const base_type& comp = *this;
            return std::equal_range(begin(), end(), key, comp);
        }

        std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
            const base_type& comp = *this;
            return std::equal_range(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            std::pair<iterator, iterator>>
        equal_range(const K& key) {
            const base_type& comp = *this;
            return std::equal_range(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            std::pair<const_iterator, const_iterator>>
        equal_range(const K& key) const {
            const base_type& comp = *this;
            return std::equal_range(begin(), end(), key, comp);
        }

        iterator lower_bound(const key_type& key) {
            const base_type& comp = *this;
            return std::lower_bound(begin(), end(), key, comp);
        }

        const_iterator lower_bound(const key_type& key) const {
            const base_type& comp = *this;
            return std::lower_bound(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            iterator>
        lower_bound(const K& key) {
            const base_type& comp = *this;
            return std::lower_bound(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const_iterator>
        lower_bound(const K& key) const {
            const base_type& comp = *this;
            return std::lower_bound(begin(), end(), key, comp);
        }

        iterator upper_bound(const key_type& key) {
            const base_type& comp = *this;
            return std::upper_bound(begin(), end(), key, comp);
        }

        const_iterator upper_bound(const key_type& key) const {
            const base_type& comp = *this;
            return std::upper_bound(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            iterator>
        upper_bound(const K& key) {
            const base_type& comp = *this;
            return std::upper_bound(begin(), end(), key, comp);
        }

        template < typename K >
        std::enable_if_t<
            detail::is_transparent_v<Compare, K>,
            const_iterator>
        upper_bound(const K& key) const {
            const base_type& comp = *this;
            return std::upper_bound(begin(), end(), key, comp);
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
            std::sort(data_.begin(), data_.end(), value_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<value_compare>(value_comp())),
                data_.end());
        }

        template < typename Iter >
        void from_range_(sorted_range_t, Iter first, Iter last) {
            assert(data_.empty());
            assert(detail::is_sorted(first, last, value_comp()));
            data_.insert(data_.end(), first, last);
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<value_compare>(value_comp())),
                data_.end());
        }

        template < typename Iter >
        void from_range_(sorted_unique_range_t, Iter first, Iter last) {
            assert(data_.empty());
            assert(detail::is_sorted_unique(first, last, value_comp()));
            data_.insert(data_.end(), first, last);
        }
    private:
        template < typename Iter >
        void insert_range_(Iter first, Iter last) {
            const auto mid_iter = data_.insert(data_.end(), first, last);
            std::sort(mid_iter, data_.end(), value_comp());
            std::inplace_merge(data_.begin(), mid_iter, data_.end(), value_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<value_compare>(value_comp())),
                data_.end());
        }

        template < typename Iter >
        void insert_range_(sorted_range_t, Iter first, Iter last) {
            assert(detail::is_sorted(first, last, value_comp()));
            const auto mid_iter = data_.insert(data_.end(), first, last);
            std::inplace_merge(data_.begin(), mid_iter, data_.end(), value_comp());
            data_.erase(
                std::unique(data_.begin(), data_.end(),
                    detail::eq_compare<value_compare>(value_comp())),
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
    flat_map(InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>>;

    template < typename InputIter, typename Allocator
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_range_t, InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>>;

    template < typename InputIter, typename Allocator
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_unique_range_t, InputIter, InputIter, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>>;

    //

    template < typename InputIter
             , typename Compare = std::less<detail::iter_key_type<InputIter>>
             , typename Allocator = std::allocator<std::pair<
                    std::add_const_t<detail::iter_key_type<InputIter>>,
                    detail::iter_mapped_type<InputIter>>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>, Compare>;

    template < typename InputIter
             , typename Compare = std::less<detail::iter_key_type<InputIter>>
             , typename Allocator = std::allocator<std::pair<
                    std::add_const_t<detail::iter_key_type<InputIter>>,
                    detail::iter_mapped_type<InputIter>>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_range_t, InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>, Compare>;

    template < typename InputIter
             , typename Compare = std::less<detail::iter_key_type<InputIter>>
             , typename Allocator = std::allocator<std::pair<
                    std::add_const_t<detail::iter_key_type<InputIter>>,
                    detail::iter_mapped_type<InputIter>>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_unique_range_t, InputIter, InputIter, Compare = Compare(), Allocator = Allocator())
        -> flat_map<detail::iter_key_type<InputIter>, detail::iter_mapped_type<InputIter>, Compare>;

    //

    template < typename Key, typename Value
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(std::initializer_list<std::pair<Key, Value>>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<std::remove_const_t<Key>, Value>;

    template < typename Key, typename Value
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_range_t, std::initializer_list<std::pair<Key, Value>>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<std::remove_const_t<Key>, Value>;

    template < typename Key, typename Value
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_unique_range_t, std::initializer_list<std::pair<Key, Value>>, Allocator, int* msvc_2017_workaround = nullptr)
        -> flat_map<std::remove_const_t<Key>, Value>;

    //

    template < typename Key, typename Value
             , typename Compare = std::less<std::remove_const_t<Key>>
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(std::initializer_list<std::pair<Key, Value>>, Compare = Compare(), Allocator = Allocator())
        -> flat_map<std::remove_const_t<Key>, Value, Compare>;

    template < typename Key, typename Value
             , typename Compare = std::less<std::remove_const_t<Key>>
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_range_t, std::initializer_list<std::pair<Key, Value>>, Compare = Compare(), Allocator = Allocator())
        -> flat_map<std::remove_const_t<Key>, Value, Compare>;

    template < typename Key, typename Value
             , typename Compare = std::less<std::remove_const_t<Key>>
             , typename Allocator = std::allocator<std::pair<const Key, Value>>
             , std::enable_if_t<!detail::is_allocator<Compare>::value, int> = 0
             , std::enable_if_t<detail::is_allocator<Allocator>::value, int> = 0 >
    flat_map(sorted_unique_range_t, std::initializer_list<std::pair<Key, Value>>, Compare = Compare(), Allocator = Allocator())
        -> flat_map<std::remove_const_t<Key>, Value, Compare>;
}

namespace flat_hpp
{
    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    void swap(
        flat_map<Key, Value, Compare, Container>& l,
        flat_map<Key, Value, Compare, Container>& r)
        noexcept(noexcept(l.swap(r)))
    {
        l.swap(r);
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator==(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return l.size() == r.size()
            && std::equal(l.begin(), l.end(), r.begin());
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator!=(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return !(l == r);
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator<(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return std::lexicographical_compare(l.begin(), l.end(), r.begin(), r.end());
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator>(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return r < l;
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator<=(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return !(r < l);
    }

    template < typename Key
             , typename Value
             , typename Compare
             , typename Container >
    bool operator>=(
        const flat_map<Key, Value, Compare, Container>& l,
        const flat_map<Key, Value, Compare, Container>& r)
    {
        return !(l < r);
    }
}
